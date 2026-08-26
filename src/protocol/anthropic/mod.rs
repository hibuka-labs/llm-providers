//! Anthropic protocol implementation.
//!
//! Implements `RawAdapter` for the Anthropic Messages API.
//! Uses `reqwest` for HTTP and `eventsource-stream` for SSE parsing,
//! with `anthropic-rs-api` types for request/response structures.

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

mod types;

use llm_trait::{
    Capabilities, CallMode, ChatRequest, ChatResponse, ChatStream, FinishReason, HttpMethod,
    ImageAttachment, LlmBackend, LlmConfig, LlmError, ProviderInfo, RawAdapter, RawRequest,
    StreamChunk, ToolCall, UsageInfo, ChatMessage,
};

use self::types::{
    ContentBlock, ContentBlockDelta, MessagesStreamEvent, StopReason,
};

/// Anthropic protocol implementation.
///
/// Handles request building, SSE stream parsing, and response conversion
/// for the Anthropic Messages API.
pub struct AnthropicProtocol {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
}

impl AnthropicProtocol {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url
                .unwrap_or("https://api.anthropic.com")
                .to_string(),
            max_tokens: 8192,
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        Self::new(
            &config.api_key,
            &config.model,
            Some(&config.base_url),
        )
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
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

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
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

        // Handle reasoning/thinking
        if let Some(ref rc) = request.reasoning {
            if rc.enabled == Some(true) || rc.budget_tokens.is_some() {
                let budget = rc.budget_tokens.unwrap_or(2048) as u32;
                body["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                });
                if budget > self.max_tokens {
                    body["max_tokens"] = Value::Number(budget.into());
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
                    system_prompt = Some(content.clone());
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
                    tool_calls,
                    ..
                } => {
                    let mut blocks = Vec::new();

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
                ChatMessage::Custom { role: _, data } => {
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "text", "text": data.to_string()}]
                    }));
                }
            }
        }

        // Merge consecutive user messages (tool_results + text)
        let mut merged: Vec<Value> = Vec::with_capacity(result.len());
        let original_count = result.len();
        let mut merge_count = 0;
        for msg in result {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            let can_merge = if let Some(prev) = merged.last() {
                let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == "user" && prev_role == "user" {
                    let prev_all_tool_results = prev
                        .get("content")
                        .and_then(|c| c.as_array())
                        .map(|blocks| {
                            blocks.iter().all(|b| {
                                b.get("type").and_then(|t| t.as_str()) == Some("tool_result")
                            })
                        })
                        .unwrap_or(false);
                    prev_all_tool_results
                } else if role == "assistant" && prev_role == "assistant" {
                    true
                } else {
                    false
                }
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
                        merge_count += 1;
                        continue;
                    }
                }
            }

            merged.push(msg);
        }

        if merge_count > 0 {
            tracing::debug!(
                original_count = original_count,
                merged_count = merged.len(),
                merge_count = merge_count,
                "messages merged"
            );
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
    fn convert_event(event: MessagesStreamEvent) -> Vec<StreamChunk> {
        match event {
            MessagesStreamEvent::MessageStart { message } => {
                vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: Some(message.usage.input_tokens),
                    completion_tokens: Some(message.usage.output_tokens),
                    total_tokens: None,
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
                _ => vec![],
            },
            MessagesStreamEvent::ContentBlockStop { .. } => vec![],
            MessagesStreamEvent::MessageDelta { delta, usage } => {
                let mut chunks = vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: None,
                    completion_tokens: Some(usage.output_tokens),
                    total_tokens: None,
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
                    chunks.push(StreamChunk::Stop {
                        finish_reason: Some(finish_reason.to_string()),
                    });
                }

                chunks
            }
            MessagesStreamEvent::MessageStop => vec![],
        }
    }

    /// Parse a non-streaming Anthropic response into ChatResponse.
    fn parse_anthropic_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        let json: Value = serde_json::from_slice(body)
            .map_err(|e| LlmError::llm(format!("Failed to parse response: {e}")))?;

        // Check for API errors
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

        // Extract content
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(blocks) = json.get("content").and_then(|c| c.as_array()) {
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            content.push_str(text);
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

        // Extract usage
        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u.get("input_tokens").and_then(|t| t.as_u64()).map(|n| n as u32),
                completion_tokens: u.get("output_tokens").and_then(|t| t.as_u64()).map(|n| n as u32),
                total_tokens: None,
            })
            .unwrap_or_default();

        // Extract finish reason
        let finish_reason = json
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "end_turn" => FinishReason::Stop,
                "tool_use" => FinishReason::ToolCalls,
                "max_tokens" => FinishReason::Length,
                other => FinishReason::Other(other.to_string()),
            })
            .unwrap_or(FinishReason::Stop);

        Ok(ChatResponse {
            content,
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

        Ok(RawRequest {
            url: format!("{}/messages", self.base_url),
            method: HttpMethod::Post,
            headers,
            body,
            stream,
        })
    }

    async fn execute_stream(
        &self,
        client: &reqwest::Client,
        request: RawRequest,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = client.post(&request.url);

        for (key, value) in &request.headers {
            builder = builder.header(key.as_str(), value.as_str());
        }

        builder = builder
            .header("Content-Type", "application/json")
            .json(&request.body);

        let response = builder
            .send()
            .await
            .map_err(|e| LlmError::llm(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            // Log full request + response on any error for debugging
            let msg_count = request.body.get("messages")
                .and_then(|m| m.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let tool_count = request.body.get("tools")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let has_thinking = request.body.get("thinking").is_some();
            tracing::error!(
                status = status.as_u16(),
                msg_count = msg_count,
                tool_count = tool_count,
                has_thinking = has_thinking,
                error_body = %body,
                request_body = %serde_json::to_string(&request.body).unwrap_or_default(),
                "API error with full request context"
            );
            return Err(LlmError::api(status.as_u16(), body));
        }

        // Parse SSE stream
        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        let chunk_stream = event_stream.flat_map(|event| match event {
            Ok(es_event) => {
                match serde_json::from_str::<MessagesStreamEvent>(&es_event.data) {
                    Ok(stream_event) => {
                        let chunks = Self::convert_event(stream_event);
                        futures_util::stream::iter(chunks.into_iter().map(Ok).collect::<Vec<_>>())
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse SSE event: {e}, data: {}", es_event.data);
                        futures_util::stream::iter(vec![])
                    }
                }
            }
            Err(e) => {
                futures_util::stream::iter(vec![Err(LlmError::stream(format!(
                    "SSE error: {e}"
                )))])
            }
        });

        Ok(ChatStream::new(Box::pin(chunk_stream)))
    }

    fn parse_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError> {
        Self::parse_anthropic_response(body)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: true,
            max_context_tokens: Some(200_000),
            max_output_tokens: Some(self.max_tokens),
        }
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "anthropic".to_string(),
            model: self.model.clone(),
            backend: LlmBackend::Anthropic,
            version: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_protocol() -> AnthropicProtocol {
        AnthropicProtocol::new("sk-test", "claude-sonnet", None)
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
    fn convert_messages_with_images() {
        let msgs = vec![ChatMessage::user_with_images(
            "pic",
            vec![
                ImageAttachment::Url {
                    url: "http://example.com/img.png".into(),
                    detail: None,
                },
                ImageAttachment::Base64 {
                    data: "abc".into(),
                    media_type: Some("image/png".into()),
                    detail: None,
                },
            ],
        )];
        let (_, out) = AnthropicProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        let blocks = out[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3); // text + 2 images
    }

    #[test]
    fn convert_messages_with_tool_calls() {
        let msgs = vec![
            ChatMessage::assistant("thinking"),
            ChatMessage::assistant_tool_call("id1", "echo", r#"{"x":1}"#),
            ChatMessage::tool("id1", "result"),
        ];
        let (_, out) = AnthropicProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[1]["role"], "user");
    }

    // ── convert_tools ──

    #[test]
    fn convert_tools_maps_correctly() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo back",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let out = AnthropicProtocol::convert_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["name"], "echo");
        assert_eq!(out[0]["description"], "echo back");
    }

    // ── build_request ──

    #[test]
    fn build_request_basic() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Stream).unwrap();
        assert!(raw.url.contains("/messages"));
        assert!(raw.stream);
        assert_eq!(raw.method, HttpMethod::Post);
        assert!(raw.headers.contains_key("x-api-key"));
        assert!(raw.headers.contains_key("anthropic-version"));
        assert_eq!(raw.body["model"], "claude-sonnet");
        assert_eq!(raw.body["stream"], true);
    }

    #[test]
    fn build_request_non_streaming() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert!(!raw.stream);
        assert_eq!(raw.body["stream"], false);
    }

    #[test]
    fn build_request_with_system() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("hi"),
        ]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["system"], "You are helpful");
    }

    #[test]
    fn build_request_with_tools() {
        let proto = make_protocol();
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo",
                "parameters": {"type": "object"}
            }
        })];
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(tools);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert!(raw.body.get("tools").is_some());
        assert_eq!(raw.body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn build_request_with_reasoning() {
        let proto = make_protocol();
        let reasoning = llm_trait::ReasoningConfig {
            enabled: Some(true),
            budget_tokens: Some(4096),
            effort: None,
        };
        let req = ChatRequest::new(vec![ChatMessage::user("think")]).with_reasoning(reasoning);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["thinking"]["type"], "enabled");
        assert_eq!(raw.body["thinking"]["budget_tokens"], 4096);
    }

    // ── parse_anthropic_response ──

    #[test]
    fn parse_response_text_only() {
        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
        )
        .unwrap();
        assert_eq!(resp.content, "Hello!");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.prompt_tokens, Some(10));
        assert_eq!(resp.usage.completion_tokens, Some(5));
    }

    #[test]
    fn parse_response_with_tool_use() {
        let body = serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}}
            ],
            "model": "claude-sonnet",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let resp = AnthropicProtocol::parse_anthropic_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
        )
        .unwrap();
        assert_eq!(resp.content, "Let me check.");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "search");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn parse_response_api_error() {
        let body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        });
        let result = AnthropicProtocol::parse_anthropic_response(
            serde_json::to_vec(&body).unwrap().as_slice(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("Invalid API key"));
    }

    // ── convert_event ──

    #[test]
    fn convert_event_text_delta() {
        let event = MessagesStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "hello".into(),
            },
        };
        let chunks = AnthropicProtocol::convert_event(event);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Text(t) if t == "hello"));
    }

    #[test]
    fn convert_event_thinking_delta() {
        let event = MessagesStreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::ThinkingDelta {
                thinking: "hmm".into(),
            },
        };
        let chunks = AnthropicProtocol::convert_event(event);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Thought(t) if t == "hmm"));
    }

    #[test]
    fn convert_event_message_delta_with_stop() {
        let event = MessagesStreamEvent::MessageDelta {
            delta: self::types::MessageDelta {
                stop_reason: Some(StopReason::EndTurn),
                stop_sequence: None,
            },
            usage: self::types::MessageDeltaUsage {
                output_tokens: 100,
                input_tokens: None,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let chunks = AnthropicProtocol::convert_event(event);
        assert_eq!(chunks.len(), 2);
        assert!(matches!(&chunks[0], StreamChunk::Usage(_)));
        assert!(
            matches!(&chunks[1], StreamChunk::Stop { finish_reason: Some(r) } if r == "end_turn")
        );
    }

    // ── capabilities / info ──

    #[test]
    fn capabilities_correct() {
        let proto = make_protocol();
        let caps = proto.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_vision);
        assert!(caps.supports_thinking);
        assert_eq!(caps.max_context_tokens, Some(200_000));
    }

    #[test]
    fn info_correct() {
        let proto = make_protocol();
        let info = proto.info();
        assert_eq!(info.name, "anthropic");
        assert_eq!(info.model, "claude-sonnet");
        assert_eq!(info.backend, LlmBackend::Anthropic);
    }

    #[test]
    fn from_config() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "claude-opus".to_string(),
            base_url: "https://custom.api.com".to_string(),
            options: Default::default(),
        };
        let proto = AnthropicProtocol::from_config(&config);
        assert_eq!(proto.model, "claude-opus");
        assert_eq!(proto.base_url, "https://custom.api.com");
    }
}
