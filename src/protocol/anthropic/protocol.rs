//! Anthropic protocol implementation.
//!
//! Implements `RawAdapter` for the Anthropic Messages API.
//! Uses `eventsource-stream` for SSE parsing,
//! with `anthropic-rs-api` types for request/response structures.

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

use llm_trait::{
    Capabilities, CallMode, ChatMessage, ChatRequest, ChatResponse, ChatStream, FinishReason,
    HttpClient, HttpMethod, ImageAttachment, LlmConfig, LlmError, ProviderInfo, RawAdapter,
    RawRequest, ReasoningSpec, StreamChunk, ToolCall, UsageInfo,
};

use super::types::{ContentBlock, ContentBlockDelta, MessagesStreamEvent, StopReason};

use crate::model_registry::ModelProfile;

/// Anthropic protocol implementation.
pub struct AnthropicProtocol {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    profile: Option<ModelProfile>,
}

impl AnthropicProtocol {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url
                .unwrap_or("https://api.anthropic.com")
                .to_string(),
            max_tokens: 16_384,  // default 16K
            profile: None,
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        let max_tokens = config
            .options
            .get("max_tokens")
            .and_then(|v| {
                v.as_str()
                    .and_then(|s| s.parse::<u32>().ok())
                    .or_else(|| v.as_u64().map(|n| n as u32))
            })
            .unwrap_or(16_384);  // default 16K

        Self::new(&config.api_key, &config.model, Some(&config.base_url))
            .with_max_tokens(max_tokens)
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set max output tokens (convenience method, overrides ModelProfile value)
    pub fn with_max_output_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_model_profile(mut self, profile: ModelProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Build Anthropic MessagesRequest from ChatRequest.
    fn build_anthropic_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<Value, LlmError> {
        let (system, messages) = Self::convert_messages(&request.messages);
        let tools = Self::convert_tools(&request.tools);

        tracing::debug!(
            model = %self.model,
            msg_count = messages.len(),
            tool_count = tools.len(),
            has_system = system.is_some(),
            has_reasoning = request.reasoning.is_some(),
            stream = stream,
            "building Anthropic request"
        );

        // Priority: user config > profile > default
        // If user explicitly set max_tokens (non-default), user config wins
        // Otherwise use profile value, fallback to default
        let default_max_tokens = 16_384;
        let effective_max_tokens = if self.max_tokens != default_max_tokens {
            // User explicitly set max_tokens, user config wins
            self.max_tokens
        } else {
            // User didn't set explicitly, use profile value or default
            self.profile
                .as_ref()
                .and_then(|p| p.capabilities.max_output_tokens)
                .unwrap_or(self.max_tokens)
        };

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": messages,
            "stream": stream,
        });

        if let Some(sys) = system {
            body["system"] = Value::String(sys);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| LlmError::llm(format!("Failed to serialize tools: {e}")))?;
            body["tool_choice"] = serde_json::json!({"type": "auto"});
        }

        // Handle reasoning: only send if profile says this model supports Thinking mode
        if let Some(ref rc) = request.reasoning {
            if let Some(ref profile) = self.profile {
                let spec = rc.to_spec(profile.reasoning_mode);
                tracing::debug!(
                    model = %self.model,
                    reasoning_mode = ?profile.reasoning_mode,
                    spec = ?spec,
                    "Anthropic reasoning config resolved"
                );
                if let ReasoningSpec::Thinking { budget_tokens } = spec {
                    let budget = budget_tokens as u32;
                    body["thinking"] = serde_json::json!({
                        "type": "enabled",
                        "budget_tokens": budget
                    });
                    if budget >= effective_max_tokens {
                        tracing::warn!(
                            budget_tokens = budget,
                            original_max_tokens = self.max_tokens,
                            "budget_tokens >= max_tokens, auto-increasing max_tokens to budget+1"
                        );
                        body["max_tokens"] = Value::Number((budget + 1).into());
                    }
                }
            }
        }

        Ok(body)
    }

    /// Convert ChatMessages to Anthropic message format.
    fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        let mut system_prompt: Option<String> = None;
        let mut result: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    match &mut system_prompt {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(content);
                        }
                        None => {
                            system_prompt = Some(content.clone());
                        }
                    }
                }
                ChatMessage::User {
                    content, images, ..
                } => {
                    let mut blocks = vec![serde_json::json!({
                        "type": "text",
                        "text": content
                    })];

                    for img in images {
                        match img {
                            ImageAttachment::Url { url, .. } => {
                                blocks.push(serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": url
                                    }
                                }));
                            }
                            ImageAttachment::Base64 {
                                data, media_type, ..
                            } => {
                                let media_type = media_type.as_deref().unwrap_or("image/jpeg");
                                blocks.push(serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data
                                    }
                                }));
                            }
                        }
                    }

                    result.push(serde_json::json!({
                        "role": "user",
                        "content": blocks
                    }));
                }
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    thinking_signature,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();

                    // Anthropic requires thinking blocks with signature to be sent back
                    // in multi-turn conversations
                    if let (Some(thinking), Some(sig)) = (&reasoning_content, &thinking_signature) {
                        blocks.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": sig
                        }));
                    }

                    if let Some(text) = content.as_ref().filter(|t| !t.is_empty()) {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }

                    if let Some(tc) = tool_calls {
                        for t in tc {
                            let input: Value = serde_json::from_str(&t.arguments)
                                .unwrap_or(Value::Object(Default::default()));
                            blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": t.id,
                                "name": t.name,
                                "input": input
                            }));
                        }
                    }

                    if !blocks.is_empty() {
                        result.push(serde_json::json!({
                            "role": "assistant",
                            "content": blocks
                        }));
                    }
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                    ..
                } => {
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content
                        }]
                    }));
                }
                ChatMessage::Custom { .. } => {
                    // P2-19: Custom messages are metadata-only, not forwarded to LLM
                }
            }
        }

        // Merge consecutive same-role messages
        let mut merged: Vec<Value> = Vec::with_capacity(result.len());
        for msg in result {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            let can_merge = if let Some(prev) = merged.last() {
                let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");
                // Merge consecutive user or assistant messages
                (role == "user" && prev_role == "user")
                    || (role == "assistant" && prev_role == "assistant")
            } else {
                false
            };

            if can_merge {
                if let Some(prev) = merged.last_mut() {
                    if let (Some(prev_content), Some(new_content)) = (
                        prev.get_mut("content").and_then(|c| c.as_array_mut()),
                        msg.get("content").and_then(|c| c.as_array()),
                    ) {
                        prev_content.extend(new_content.iter().cloned());
                        continue;
                    }
                }
            }

            merged.push(msg);
        }

        (system_prompt, merged)
    }

    /// Convert OpenAI-style tools to Anthropic tool format.
    fn convert_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .filter_map(|tool| {
                let func = tool.get("function")?;
                let name = func.get("name")?.as_str()?;
                let description = func
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let input_schema = func
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object"}));
                Some(serde_json::json!({
                    "name": name,
                    "description": description,
                    "input_schema": input_schema
                }))
            })
            .collect()
    }

    /// Convert an Anthropic stream event to StreamChunk(s).
    /// Returns Err for error events (propagated as stream-level errors).
    fn convert_event(event: MessagesStreamEvent) -> Result<Vec<StreamChunk>, LlmError> {
        let chunks = match event {
            MessagesStreamEvent::MessageStart { message } => {
                vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: Some(message.usage.input_tokens),
                    completion_tokens: Some(message.usage.output_tokens),
                    total_tokens: None,
                    reasoning_tokens: None,
                })]
            }
            MessagesStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ContentBlock::ToolUse { id, name, .. } => {
                    vec![StreamChunk::ToolCall(serde_json::json!({
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "function": {
                                    "name": name,
                                    "arguments": "",
                                }
                            }]
                        }
                    }))]
                }
                _ => vec![],
            },
            MessagesStreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentBlockDelta::TextDelta { text } => {
                    vec![StreamChunk::Text(text)]
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    vec![StreamChunk::ToolCall(serde_json::json!({
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {
                                    "arguments": partial_json,
                                }
                            }]
                        }
                    }))]
                }
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    vec![StreamChunk::Thought(thinking)]
                }
                ContentBlockDelta::SignatureDelta { signature } => {
                    vec![StreamChunk::ThinkingSignature(signature)]
                }
                _ => vec![],
            },
            MessagesStreamEvent::ContentBlockStop { .. } => vec![],
            MessagesStreamEvent::MessageDelta { delta, usage } => {
                let mut chunks = vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: None,
                    completion_tokens: Some(usage.output_tokens),
                    total_tokens: None,
                    reasoning_tokens: None,
                })];

                if let Some(reason) = delta.stop_reason {
                    let finish_reason = match reason {
                        StopReason::EndTurn => "end_turn",
                        StopReason::ToolUse => "tool_use",
                        StopReason::MaxTokens => "max_tokens",
                        StopReason::StopSequence => "stop_sequence",
                        StopReason::PauseTurn => "pause_turn",
                        StopReason::Refusal => "refusal",
                    };
                    if reason == StopReason::MaxTokens {
                        tracing::warn!(
                            finish_reason,
                            "stream truncated by token limit — tool call arguments may be incomplete"
                        );
                    }
                    chunks.push(StreamChunk::Stop {
                        finish_reason: Some(finish_reason.to_string()),
                    });
                }

                chunks
            }
            MessagesStreamEvent::MessageStop => vec![],
            MessagesStreamEvent::Ping => vec![],
            MessagesStreamEvent::Error { error } => {
                tracing::error!(
                    error_type = %error.error_type,
                    message = %error.message,
                    "Anthropic stream error event"
                );
                return Err(LlmError::llm(format!(
                    "{}: {}",
                    error.error_type, error.message
                )));
            }
            MessagesStreamEvent::Unknown => {
                tracing::debug!("Ignoring unknown Anthropic-protocol SSE event (provider extensions like DeepSeek ping)");
                vec![]
            }
        };
        Ok(chunks)
    }

    /// Parse a non-streaming Anthropic response into ChatResponse.
    fn parse_anthropic_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        let json: Value = serde_json::from_slice(body)
            .map_err(|e| LlmError::llm(format!("Failed to parse response: {e}")))?;

        if let Some(error) = json.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            let err_type = error
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("api_error");
            return Err(LlmError::llm(format!("{err_type}: {msg}")));
        }

        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut thinking_signature = None;
        let mut tool_calls = Vec::new();

        if let Some(blocks) = json.get("content").and_then(|c| c.as_array()) {
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            content.push_str(text);
                        }
                    }
                    Some("thinking") => {
                        if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                            reasoning_content.push_str(thinking);
                        }
                        // Extract signature for multi-turn thinking
                        if let Some(sig) = block.get("signature").and_then(|s| s.as_str()) {
                            thinking_signature = Some(sig.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = block
                            .get("input")
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u
                    .get("input_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                completion_tokens: u
                    .get("output_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                total_tokens: None,
                reasoning_tokens: None,
            })
            .unwrap_or_default();

        let finish_reason = json
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "end_turn" => FinishReason::Stop,
                "tool_use" => FinishReason::ToolCalls,
                "max_tokens" => FinishReason::Length,
                "stop_sequence" => FinishReason::Stop,
                "refusal" => FinishReason::ContentFilter,
                "pause_turn" => FinishReason::Stop,
                other => FinishReason::Other(other.to_string()),
            })
            .unwrap_or(FinishReason::Stop);

        Ok(ChatResponse {
            content,
            reasoning_content: if reasoning_content.is_empty() {
                None
            } else {
                Some(reasoning_content)
            },
            thinking_signature,
            tool_calls,
            usage,
            finish_reason,
            raw: Some(json),
        })
    }
}

#[async_trait]
impl RawAdapter for AnthropicProtocol {
    fn build_request(
        &self,
        request: &ChatRequest,
        mode: CallMode,
    ) -> Result<RawRequest, LlmError> {
        let stream = mode == CallMode::Stream;
        let body = self.build_anthropic_request(request, stream)?;

        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), self.api_key.clone());
        headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());

        // Build URL: append /v1/messages if base_url doesn't already end with /v1
        let base = self.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{}/messages", base)
        } else {
            format!("{}/v1/messages", base)
        };

        Ok(RawRequest {
            url,
            method: HttpMethod::Post,
            headers,
            body,
            stream,
        })
    }

    async fn execute_stream(
        &self,
        client: &dyn HttpClient,
        request: RawRequest,
    ) -> Result<ChatStream, LlmError> {
        let response = client.send(&request).await?;

        if !response.is_success() {
            let status = response.status();
            let body = response.text().await;
            tracing::error!(
                status = status,
                error_body = %body,
                request_body = %serde_json::to_string(&request.body).unwrap_or_default(),
                "Anthropic API error with full request context"
            );
            return Err(LlmError::api(status, body));
        }

        self.parse_sse_stream(client, request, response).await
    }

    async fn parse_sse_stream(
        &self,
        _client: &dyn HttpClient,
        _request: RawRequest,
        response: llm_trait::HttpResponse,
    ) -> Result<ChatStream, LlmError> {
        // Parse SSE stream
        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        let chunk_stream = event_stream.flat_map(|event| match event {
            Ok(es_event) => {
                match serde_json::from_str::<MessagesStreamEvent>(&es_event.data) {
                    Ok(stream_event) => {
                        match Self::convert_event(stream_event) {
                            Ok(chunks) => {
                                futures_util::stream::iter(chunks.into_iter().map(Ok).collect::<Vec<_>>())
                            }
                            Err(e) => {
                                futures_util::stream::iter(vec![Err(e)])
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse SSE event: {e}, data: {}", es_event.data);
                        // Propagate parse error instead of swallowing
                        futures_util::stream::iter(vec![Err(LlmError::stream(format!(
                            "Failed to parse SSE event: {e}"
                        )))])
                    }
                }
            }
            Err(e) => futures_util::stream::iter(vec![Err(LlmError::stream(format!(
                "SSE error: {e}"
            )))]),
        });

        Ok(ChatStream::new(Box::pin(chunk_stream)))
    }

    fn parse_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError> {
        Self::parse_anthropic_response(body)
    }

    fn capabilities(&self) -> Capabilities {
        self.profile
            .as_ref()
            .map(|p| p.capabilities.clone())
            .unwrap_or_else(|| Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: true,
                supports_thinking: true,
                max_context_tokens: Some(200_000),
                max_output_tokens: Some(self.max_tokens),
            })
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: self
                .profile
                .as_ref()
                .map(|p| p.provider_name.to_string())
                .unwrap_or_else(|| "anthropic".to_string()),
            model: self.model.clone(),
            version: None,
        }
    }
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    use super::*;

    pub fn parse_anthropic_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        AnthropicProtocol::parse_anthropic_response(body)
    }

    pub fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        AnthropicProtocol::convert_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_trait::{Protocol, ReasoningConfig, ReasoningMode};
    use crate::protocol::anthropic::types::AnthropicError;

    fn make_protocol() -> AnthropicProtocol {
        AnthropicProtocol::new("sk-test", "claude-sonnet", None)
    }

    fn make_protocol_with_profile(profile: ModelProfile) -> AnthropicProtocol {
        AnthropicProtocol::new("sk-test", "claude-sonnet", None).with_model_profile(profile)
    }

    fn claude_profile() -> ModelProfile {
        ModelProfile {
            protocol: Protocol::Anthropic,
            provider_name: "anthropic",
            capabilities: Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: true,
                supports_thinking: true,
                max_context_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                ..Default::default()
            },
            reasoning_mode: ReasoningMode::Thinking,
            supported_extra_params: &[],
        }
    }

    // ── Claude: thinking block IS sent ──

    #[test]
    fn claude_sends_thinking_block() {
        let proto = make_protocol_with_profile(claude_profile());
        let reasoning = ReasoningConfig {
            budget_tokens: Some(4096),
            ..Default::default()
        };
        let req = ChatRequest::new(vec![ChatMessage::user("think")]).with_reasoning(reasoning);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["thinking"]["type"], "enabled");
        assert_eq!(raw.body["thinking"]["budget_tokens"], 4096);
    }

    // ── Non-streaming reasoning_content ──

    #[test]
    fn parse_response_with_thinking_block() {
        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "The answer is 42."}
            ],
            "model": "claude-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
        )
        .unwrap();
        assert_eq!(resp.content, "The answer is 42.");
        assert_eq!(resp.reasoning_content.as_deref(), Some("Let me reason..."));
        // No signature in this response
        assert!(resp.thinking_signature.is_none());
    }

    #[test]
    fn parse_response_with_thinking_block_and_signature() {
        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason...", "signature": "sig_abc123"},
                {"type": "text", "text": "The answer is 42."}
            ],
            "model": "claude-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
        )
        .unwrap();
        assert_eq!(resp.content, "The answer is 42.");
        assert_eq!(resp.reasoning_content.as_deref(), Some("Let me reason..."));
        // Signature should be extracted
        assert_eq!(resp.thinking_signature.as_deref(), Some("sig_abc123"));
    }

    // ── convert_messages ──

    #[test]
    fn convert_messages_basic() {
        let msgs = vec![ChatMessage::user("hi")];
        let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
        assert!(sys.is_none());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
    }

    #[test]
    fn convert_messages_with_system() {
        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
        assert_eq!(sys.as_deref(), Some("sys"));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn convert_messages_multiple_system_messages_merged() {
        // P2-15: Multiple System messages should be merged, not just keep the last one.
        let msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::system("Always respond in Chinese."),
            ChatMessage::user("hi"),
        ];
        let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
        let system = sys.expect("should have system prompt");
        assert!(
            system.contains("helpful assistant"),
            "P2-15: first system message should be preserved"
        );
        assert!(
            system.contains("Chinese"),
            "P2-15: second system message should be preserved"
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn convert_messages_assistant_reasoning_content_preserved() {
        // P0-2 FIXED: convert_messages now preserves reasoning_content as a thinking block
        // when thinking_signature is present.
        let msgs = vec![
            ChatMessage::user("think about this"),
            ChatMessage::Assistant {
                content: Some("the answer".to_string()),
                reasoning_content: Some("let me reason step by step...".to_string()),
                thinking_signature: Some("sig_abc123".to_string()),
                tool_calls: None,
            },
            ChatMessage::user("follow up"),
        ];
        let (_sys, out) = AnthropicProtocol::convert_messages(&msgs);
        let assistant_msg = out.iter().find(|m| m["role"] == "assistant").unwrap();
        let blocks = assistant_msg["content"].as_array().unwrap();

        // Should have thinking block + text block
        assert_eq!(blocks.len(), 2);

        // First block should be thinking with signature
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "let me reason step by step...");
        assert_eq!(blocks[0]["signature"], "sig_abc123");

        // Second block should be text
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "the answer");
    }

    #[test]
    fn convert_messages_assistant_no_thinking_without_signature() {
        // When reasoning_content exists but thinking_signature is None,
        // no thinking block should be generated (signature is required by Anthropic)
        let msgs = vec![
            ChatMessage::user("think about this"),
            ChatMessage::assistant_with_reasoning("the answer", "let me reason..."),
            ChatMessage::user("follow up"),
        ];
        let (_sys, out) = AnthropicProtocol::convert_messages(&msgs);
        let assistant_msg = out.iter().find(|m| m["role"] == "assistant").unwrap();
        let blocks = assistant_msg["content"].as_array().unwrap();

        // Should only have text block (no thinking without signature)
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
    }

    // ── build_request ──

    #[test]
    fn build_request_basic() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Stream).unwrap();
        // URL should be {base_url}/v1/messages (Anthropic standard path)
        assert!(
            raw.url.ends_with("/v1/messages"),
            "URL should end with /v1/messages, got: {}",
            raw.url
        );
        assert!(raw.stream);
        assert_eq!(raw.method, HttpMethod::Post);
        assert!(raw.headers.contains_key("x-api-key"));
        assert!(raw.headers.contains_key("anthropic-version"));
        assert_eq!(raw.body["model"], "claude-sonnet");
    }

    #[test]
    fn build_request_url_with_v1_suffix() {
        // When base_url already ends with /v1, don't double it
        let proto = AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1"));
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.url, "https://api.example.com/v1/messages");
    }

    #[test]
    fn build_request_url_without_v1_suffix() {
        // When base_url doesn't end with /v1, add it
        let proto = AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com"));
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.url, "https://api.example.com/v1/messages");
    }

    #[test]
    fn build_request_url_with_trailing_slash() {
        // Trailing slash should not cause double-slash in URL
        let proto = AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1/"));
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert!(
            !raw.url.contains("//v1"),
            "URL should not have double-slash, got: {}",
            raw.url
        );
        assert!(
            raw.url.ends_with("/v1/messages"),
            "URL should end with /v1/messages, got: {}",
            raw.url
        );
    }

    // ── info ──

    #[test]
    fn convert_event_error_returns_err() {
        // P1-4/Finding #12: Error events should return Err (stream-level error),
        // consistent with OpenAI's error handling.
        let event = MessagesStreamEvent::Error {
            error: AnthropicError {
                error_type: "overloaded_error".to_string(),
                message: "Too many requests".to_string(),
            },
        };
        let result = AnthropicProtocol::convert_event(event);
        assert!(result.is_err(), "Error event should return Err");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("overloaded_error"), "Error should contain type");
        assert!(err_msg.contains("Too many requests"), "Error should contain message");
    }

    #[test]
    fn info_with_profile() {
        let proto = make_protocol_with_profile(claude_profile());
        let info = proto.info();
        assert_eq!(info.name, "anthropic");
    }

    #[test]
    fn info_without_profile() {
        let proto = make_protocol();
        let info = proto.info();
        assert_eq!(info.name, "anthropic");
    }

    #[test]
    fn build_request_budget_equals_max_tokens_must_increase() {
        // P2-14: Anthropic requires max_tokens > budget_tokens.
        // When budget == max_tokens, we must increase max_tokens.
        let profile = ModelProfile {
            protocol: Protocol::Anthropic,
            provider_name: "anthropic",
            capabilities: Capabilities {
                max_output_tokens: Some(8_192),
                ..claude_profile().capabilities
            },
            reasoning_mode: ReasoningMode::Thinking,
            supported_extra_params: &[],
        };
        let proto = make_protocol_with_profile(profile);
        let reasoning = ReasoningConfig {
            budget_tokens: Some(8_192), // == max_tokens
            ..Default::default()
        };
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        let max_tokens = raw.body["max_tokens"].as_u64().unwrap();
        let budget = raw.body["thinking"]["budget_tokens"].as_u64().unwrap();

        assert!(
            max_tokens > budget,
            "P2-14: max_tokens ({max_tokens}) must be > budget_tokens ({budget}), Anthropic rejects equal values"
        );
    }

    #[test]
    fn build_request_uses_profile_max_output_tokens() {
        // P1-6 FIXED: AnthropicProtocol now uses profile.max_output_tokens.
        let profile = ModelProfile {
            protocol: Protocol::Anthropic,
            provider_name: "anthropic",
            capabilities: Capabilities {
                max_output_tokens: Some(64_000),
                ..claude_profile().capabilities
            },
            reasoning_mode: ReasoningMode::Thinking,
            supported_extra_params: &[],
        };
        let proto = make_protocol_with_profile(profile);
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        let max_tokens = raw.body["max_tokens"].as_u64().unwrap();

        assert_eq!(
            max_tokens, 64_000,
            "P1-6 FIXED: Anthropic should use profile.max_output_tokens"
        );
    }

    // ── convert_tools ──

    #[test]
    fn convert_tools_basic() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }
        })];
        let result = AnthropicProtocol::convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["name"], "search");
        assert_eq!(result[0]["description"], "Search the web");
        assert_eq!(result[0]["input_schema"]["type"], "object");
    }

    #[test]
    fn convert_tools_missing_description_and_params() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": { "name": "minimal" }
        })];
        let result = AnthropicProtocol::convert_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["description"], "");
        assert_eq!(result[0]["input_schema"]["type"], "object");
    }

    #[test]
    fn convert_tools_missing_function_key() {
        let tools = vec![serde_json::json!({"type": "function"})];
        let result = AnthropicProtocol::convert_tools(&tools);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn convert_tools_missing_name() {
        let tools = vec![serde_json::json!({
            "function": { "description": "no name" }
        })];
        let result = AnthropicProtocol::convert_tools(&tools);
        assert_eq!(result.len(), 0);
    }

    // ── parse_anthropic_response: error paths ──

    #[test]
    fn parse_response_invalid_json() {
        let result = AnthropicProtocol::parse_anthropic_response(b"not json");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));
    }

    #[test]
    fn parse_response_error_object() {
        let body = serde_json::json!({
            "error": {"message": "rate limited", "type": "rate_error"}
        });
        let result = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rate_error: rate limited"));
    }

    #[test]
    fn parse_response_error_missing_fields() {
        let body = serde_json::json!({"error": {}});
        let result = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("api_error: unknown error"));
    }

    // ── parse_anthropic_response: stop_reason variants ──

    #[test]
    fn parse_response_stop_reason_max_tokens() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "truncated"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    #[test]
    fn parse_response_stop_reason_stop_sequence() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn parse_response_stop_reason_refusal() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "refused"}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::ContentFilter);
    }

    #[test]
    fn parse_response_stop_reason_unknown() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "something_new",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Other("something_new".into()));
    }

    #[test]
    fn parse_response_no_stop_reason_defaults_to_stop() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    // ── parse_anthropic_response: missing content/usage ──

    #[test]
    fn parse_response_no_content_array() {
        let body = serde_json::json!({
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.content, "");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn parse_response_no_usage() {
        let body = serde_json::json!({
            "content": [{"type": "text", "text": "x"}]
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.usage.prompt_tokens, None);
    }

    // ── parse_anthropic_response: tool_use with missing fields ──

    #[test]
    fn parse_response_tool_use_missing_fields() {
        let body = serde_json::json!({
            "content": [
                {"type": "tool_use"},
                {"type": "tool_use", "id": "t1"},
                {"type": "tool_use", "id": "t2", "name": "fn"},
                {"type": "tool_use", "id": "t3", "name": "fn2", "input": {"x": 1}}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.tool_calls.len(), 4);
        assert_eq!(resp.tool_calls[0].id, "");
        assert_eq!(resp.tool_calls[0].name, "");
        assert_eq!(resp.tool_calls[0].arguments, "{}");
        assert_eq!(resp.tool_calls[1].name, "");
        assert_eq!(resp.tool_calls[2].arguments, "{}");
        assert_eq!(resp.tool_calls[3].arguments, r#"{"x":1}"#);
    }

    // ── parse_anthropic_response: thinking block ──

    #[test]
    fn parse_response_thinking_block_with_signature() {
        let body = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "let me think...", "signature": "sig123"},
                {"type": "text", "text": "answer"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("let me think..."));
        assert_eq!(resp.thinking_signature.as_deref(), Some("sig123"));
        assert_eq!(resp.content, "answer");
    }

    #[test]
    fn parse_response_thinking_block_no_signature() {
        let body = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "ok"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("hmm"));
        assert_eq!(resp.thinking_signature, None);
    }

    // ── convert_messages: tool message ──

    #[test]
    fn convert_messages_tool_role() {
        let msgs = vec![ChatMessage::tool("call_1", "result data")];
        let (_, out) = AnthropicProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_1");
        assert_eq!(content[0]["content"], "result data");
    }

    // ── convert_messages: custom filtered ──

    #[test]
    fn convert_messages_custom_filtered() {
        let msgs = vec![
            ChatMessage::user("before"),
            ChatMessage::Custom { role: "custom".into(), data: serde_json::json!({"secret": true}) },
            ChatMessage::user("after"),
        ];
        let (_, out) = AnthropicProtocol::convert_messages(&msgs);
        // Custom filtered, two users merged
        assert_eq!(out.len(), 1);
        let content = out[0]["content"].as_array().unwrap();
        let text: String = content.iter().map(|c| c["text"].as_str().unwrap_or("")).collect();
        assert!(!text.contains("secret"));
    }

    // ── convert_messages: assistant with empty content ──

    #[test]
    fn convert_messages_assistant_empty_content() {
        let msgs = vec![ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: None,
            thinking_signature: None,
        }];
        let (_, out) = AnthropicProtocol::convert_messages(&msgs);
        // Empty assistant produces empty blocks array, filtered out by `!blocks.is_empty()`
        assert_eq!(out.len(), 0);
    }

    // ── capabilities ──

    #[test]
    fn capabilities_with_profile() {
        let profile = ModelProfile {
            protocol: Protocol::Anthropic,
            provider_name: "anthropic",
            capabilities: Capabilities {
                supports_vision: true,
                supports_thinking: true,
                ..Default::default()
            },
            reasoning_mode: ReasoningMode::Thinking,
            supported_extra_params: &[],
        };
        let proto = make_protocol_with_profile(profile);
        let caps = proto.capabilities();
        assert!(caps.supports_vision);
        assert!(caps.supports_thinking);
    }

    #[test]
    fn capabilities_without_profile() {
        let proto = make_protocol();
        let caps = proto.capabilities();
        // Default Capabilities has supports_vision=true for Anthropic
        assert!(caps.supports_vision);
    }

    // ── from_config ──

    #[test]
    fn from_config_with_string_max_tokens() {
        use llm_trait::LlmConfig;
        let mut options = std::collections::HashMap::new();
        options.insert("max_tokens".into(), serde_json::json!("4096"));
        let config = LlmConfig {
            api_key: "sk-test".into(),
            model: "claude-sonnet".into(),
            base_url: "https://api.anthropic.com".into(),
            options,
            ..Default::default()
        };
        let proto = AnthropicProtocol::from_config(&config);
        assert_eq!(proto.max_tokens, 4096);
    }

    #[test]
    fn from_config_with_numeric_max_tokens() {
        use llm_trait::LlmConfig;
        let mut options = std::collections::HashMap::new();
        options.insert("max_tokens".into(), serde_json::json!(16384));
        let config = LlmConfig {
            api_key: "sk-test".into(),
            model: "claude-sonnet".into(),
            base_url: "https://api.anthropic.com".into(),
            options,
            ..Default::default()
        };
        let proto = AnthropicProtocol::from_config(&config);
        assert_eq!(proto.max_tokens, 16384);
    }

    #[test]
    fn from_config_default_max_tokens() {
        use llm_trait::LlmConfig;
        let config = LlmConfig {
            api_key: "sk-test".into(),
            model: "claude-sonnet".into(),
            base_url: "https://api.anthropic.com".into(),
            ..Default::default()
        };
        let proto = AnthropicProtocol::from_config(&config);
        assert_eq!(proto.max_tokens, 16384);
    }

    // ── build_request: system message ──

    #[test]
    fn build_request_with_system_message() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("hello"),
        ]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["system"], "You are a helpful assistant.");
    }

    // ── build_request: tools ──

    #[test]
    fn build_request_with_tools() {
        let proto = make_protocol();
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }
        })];
        let req = ChatRequest::new(vec![ChatMessage::user("search for cats")])
            .with_tools(tools);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert!(raw.body["tools"].is_array());
        assert_eq!(raw.body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(raw.body["tool_choice"]["type"], "auto");
    }

    // ── proptest: convert_messages ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a random ChatMessage (simplified: system, user, assistant, tool).
        fn arb_chat_message() -> impl Strategy<Value = ChatMessage> {
            prop_oneof![
                "[a-z ]{0,50}".prop_map(|s| ChatMessage::system(&s)),
                "[a-z ]{0,50}".prop_map(|s| ChatMessage::user(&s)),
                "[a-z ]{0,50}".prop_map(|s| ChatMessage::assistant(&s)),
                ("[a-z]{1,10}", "[a-z ]{0,50}").prop_map(|(id, content)| ChatMessage::tool(&id, &content)),
            ]
        }

        proptest! {
            #[test]
            fn convert_messages_output_len_le_input_len(
                messages in prop::collection::vec(arb_chat_message(), 0..20)
            ) {
                let (_, out) = AnthropicProtocol::convert_messages(&messages);
                // System messages extracted → output ≤ non-system count
                let non_system_count = messages.iter()
                    .filter(|m| !matches!(m, ChatMessage::System { .. }))
                    .filter(|m| !matches!(m, ChatMessage::Custom { .. }))
                    .count();
                assert!(out.len() <= non_system_count,
                    "output {} > non-system input {}", out.len(), non_system_count);
            }

            #[test]
            fn convert_messages_output_roles_are_valid(
                messages in prop::collection::vec(arb_chat_message(), 0..20)
            ) {
                let (_, out) = AnthropicProtocol::convert_messages(&messages);
                let valid_roles = ["user", "assistant"];
                for msg in &out {
                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    assert!(valid_roles.contains(&role),
                        "invalid role: {:?}, full message: {:?}", role, msg);
                }
            }

            #[test]
            fn convert_messages_system_extracted(
                content in "[a-z ]{1,50}"
            ) {
                let msgs = vec![ChatMessage::system(&content)];
                let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
                assert!(sys.is_some(), "system message should be extracted");
                assert!(out.is_empty(), "system message should not appear in output");
            }
        }
    }
}
