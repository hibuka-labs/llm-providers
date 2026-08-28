//! OpenAI Chat Completions protocol implementation.
//!
//! Implements `RawAdapter` for the OpenAI Chat Completions API.
//! Also serves as the base protocol for OpenAI-compatible providers
//! (DeepSeek, Qwen, Azure, local models, etc.).

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

use llm_trait::{
    Capabilities, CallMode, ChatMessage, ChatRequest, ChatResponse, ChatStream, FinishReason,
    HttpClient, HttpMethod, ImageAttachment, ImageDetail, LlmConfig, LlmError, ProviderInfo,
    RawAdapter, RawRequest, ReasoningSpec, StreamChunk, ToolCall, UsageInfo,
};

use crate::model_registry::ModelProfile;

/// OpenAI Chat Completions protocol implementation.
///
/// Handles request building, SSE stream parsing, and response conversion
/// for the OpenAI Chat Completions API (`/v1/chat/completions`).
pub struct OpenAiProtocol {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
    profile: Option<ModelProfile>,
}

impl OpenAiProtocol {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url
                .unwrap_or("https://api.openai.com/v1")
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

    /// Build OpenAI Chat Completions request body from ChatRequest.
    fn build_openai_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<Value, LlmError> {
        let messages = Self::convert_messages(&request.messages);
        let tools = Self::convert_tools(&request.tools);

        tracing::debug!(
            model = %self.model,
            msg_count = messages.len(),
            tool_count = tools.len(),
            has_reasoning = request.reasoning.is_some(),
            stream = stream,
            "building OpenAI request"
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

        tracing::debug!(
            model = %self.model,
            effective_max_tokens,
            configured_max_tokens = self.max_tokens,
            has_profile = self.profile.is_some(),
            "OpenAI request max_tokens resolved"
        );

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": effective_max_tokens,
        });

        // P2-18: Request usage data in stream mode
        if stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| LlmError::llm(format!("Failed to serialize tools: {e}")))?;
        }

        // Handle reasoning: only send if profile says this model supports Effort mode
        if let Some(ref rc) = request.reasoning {
            if let Some(ref profile) = self.profile {
                let spec = rc.to_spec(profile.reasoning_mode);
                tracing::debug!(
                    model = %self.model,
                    reasoning_mode = ?profile.reasoning_mode,
                    spec = ?spec,
                    "reasoning config resolved"
                );
                if let ReasoningSpec::Effort(effort) = spec {
                    let effort_str = match effort {
                        llm_trait::ReasoningEffort::Low => "low",
                        llm_trait::ReasoningEffort::Medium => "medium",
                        llm_trait::ReasoningEffort::High => "high",
                        // ReasoningEffort::None is filtered out by to_spec(),
                        // so ReasoningSpec::Effort(None) never reaches here.
                        llm_trait::ReasoningEffort::None => unreachable!(),
                        llm_trait::ReasoningEffort::XHigh => "high",
                    };
                    body["reasoning_effort"] = Value::String(effort_str.to_string());
                }
            }
        }

        Ok(body)
    }

    /// Convert ChatMessages to OpenAI message format.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    result.push(serde_json::json!({
                        "role": "system",
                        "content": content
                    }));
                }
                ChatMessage::User {
                    content, images, ..
                } => {
                    if images.is_empty() {
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content
                        }));
                    } else {
                        let mut parts = vec![serde_json::json!({
                            "type": "text",
                            "text": content
                        })];

                        for img in images {
                            match img {
                                ImageAttachment::Url { url, detail } => {
                                    let mut image_part = serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": url }
                                    });
                                    if let Some(d) = detail {
                                        let detail_str = match d {
                                            ImageDetail::Low => "low",
                                            ImageDetail::High => "high",
                                            ImageDetail::Auto => "auto",
                                        };
                                        image_part["image_url"]["detail"] =
                                            Value::String(detail_str.to_string());
                                    }
                                    parts.push(image_part);
                                }
                                ImageAttachment::Base64 {
                                    data,
                                    media_type,
                                    detail,
                                } => {
                                    let media_type = media_type.as_deref().unwrap_or("image/jpeg");
                                    let url = format!("data:{media_type};base64,{data}");
                                    let mut image_part = serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": url }
                                    });
                                    if let Some(d) = detail {
                                        let detail_str = match d {
                                            ImageDetail::Low => "low",
                                            ImageDetail::High => "high",
                                            ImageDetail::Auto => "auto",
                                        };
                                        image_part["image_url"]["detail"] =
                                            Value::String(detail_str.to_string());
                                    }
                                    parts.push(image_part);
                                }
                            }
                        }

                        result.push(serde_json::json!({
                            "role": "user",
                            "content": parts
                        }));
                    }
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    let mut msg = serde_json::json!({
                        "role": "assistant"
                    });

                    if let Some(text) = content.as_ref().filter(|t| !t.is_empty()) {
                        msg["content"] = Value::String(text.clone());
                    }

                    if let Some(tc) = tool_calls {
                        let openai_tool_calls: Vec<Value> = tc
                            .iter()
                            .map(|t| {
                                serde_json::json!({
                                    "id": t.id,
                                    "type": "function",
                                    "function": {
                                        "name": t.name,
                                        "arguments": t.arguments
                                    }
                                })
                            })
                            .collect();
                        msg["tool_calls"] = Value::Array(openai_tool_calls);
                    }

                    result.push(msg);
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                    ..
                } => {
                    result.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content
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
                role == prev_role && role != "tool"
            } else {
                false
            };

            if can_merge {
                if let Some(prev) = merged.last_mut() {
                    if role == "assistant" {
                        let prev_has_tc = prev.get("tool_calls").map_or(false, |t| t.is_array());
                        let new_has_tc = msg.get("tool_calls").map_or(false, |t| t.is_array());

                        // P2-20: Don't merge tool_calls+text into one message.
                        // OpenAI API requires tool_calls messages to have null/empty content.
                        if prev_has_tc != new_has_tc {
                            // One has tool_calls, the other has text — don't merge
                            merged.push(msg);
                            continue;
                        }

                        let prev_content =
                            prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let new_content =
                            msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let combined_content = if prev_content.is_empty() {
                            new_content.to_string()
                        } else if new_content.is_empty() {
                            prev_content.to_string()
                        } else {
                            format!("{}\n{}", prev_content, new_content)
                        };
                        if !combined_content.is_empty() {
                            prev["content"] = Value::String(combined_content);
                        }

                        if let Some(new_tc) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                            if let Some(prev_tc) =
                                prev.get_mut("tool_calls").and_then(|t| t.as_array_mut())
                            {
                                prev_tc.extend(new_tc.iter().cloned());
                            } else {
                                prev["tool_calls"] = Value::Array(new_tc.clone());
                            }
                        }
                        continue;
                    }
                    if role == "user" {
                        let prev_is_array =
                            prev.get("content").map_or(false, |c| c.is_array());
                        let new_is_array =
                            msg.get("content").map_or(false, |c| c.is_array());

                        if prev_is_array || new_is_array {
                            // Either side has array content (images) — merge as arrays
                            let mut prev_parts: Vec<Value> = prev
                                .get("content")
                                .and_then(|c| c.as_array())
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Convert string content to a text part
                                    prev.get("content")
                                        .and_then(|c| c.as_str())
                                        .filter(|s| !s.is_empty())
                                        .map(|s| {
                                            vec![serde_json::json!({"type": "text", "text": s})]
                                        })
                                        .unwrap_or_default()
                                });
                            let new_parts: Vec<Value> = msg
                                .get("content")
                                .and_then(|c| c.as_array())
                                .cloned()
                                .unwrap_or_else(|| {
                                    msg.get("content")
                                        .and_then(|c| c.as_str())
                                        .filter(|s| !s.is_empty())
                                        .map(|s| {
                                            vec![serde_json::json!({"type": "text", "text": s})]
                                        })
                                        .unwrap_or_default()
                                });
                            prev_parts.extend(new_parts);
                            prev["content"] = Value::Array(prev_parts);
                        } else {
                            // Both are strings — merge as strings
                            let prev_content =
                                prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                            let new_content =
                                msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                            let combined = if prev_content.is_empty() {
                                new_content.to_string()
                            } else if new_content.is_empty() {
                                prev_content.to_string()
                            } else {
                                format!("{}\n{}", prev_content, new_content)
                            };
                            prev["content"] = Value::String(combined);
                        }
                        continue;
                    }
                }
            }

            merged.push(msg);
        }

        merged
    }

    /// Convert tools to OpenAI format.
    fn convert_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .map(|tool| {
                if tool.get("type").is_some() && tool.get("function").is_some() {
                    return tool.clone();
                }
                serde_json::json!({
                    "type": "function",
                    "function": tool
                })
            })
            .collect()
    }

    /// Parse a non-streaming OpenAI response into ChatResponse.
    fn parse_openai_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
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

        let choice = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| LlmError::llm("No choices in response"))?;

        let message = choice
            .get("message")
            .ok_or_else(|| LlmError::llm("No message in choice"))?;

        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Extract reasoning_content (for o1/o3-style and DeepSeek models)
        let reasoning_content = message
            .get("reasoning_content")
            .and_then(|r| r.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let tool_calls = message
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let id = tc.get("id")?.as_str()?.to_string();
                        let function = tc.get("function")?;
                        let name = function.get("name")?.as_str()?.to_string();
                        let arguments = function
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        Some(ToolCall { id, name, arguments })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u
                    .get("prompt_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                completion_tokens: u
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                total_tokens: u
                    .get("total_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
            })
            .unwrap_or_default();

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            })
            .unwrap_or(FinishReason::Stop);

        Ok(ChatResponse {
            content,
            reasoning_content,
            thinking_signature: None, // OpenAI doesn't use thinking signatures
            tool_calls,
            usage,
            finish_reason,
            raw: Some(json),
        })
    }

    /// Convert a single OpenAI SSE delta to StreamChunk(s).
    fn convert_delta(delta: &Value, index: usize) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();

        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                chunks.push(StreamChunk::Text(content.to_string()));
            }
        }

        if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
            if !reasoning.is_empty() {
                chunks.push(StreamChunk::Thought(reasoning.to_string()));
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let tc_index = tc
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .unwrap_or(index as u64) as usize;
                let mut delta_obj = serde_json::json!({});

                if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                    delta_obj["id"] = Value::String(id.to_string());
                }

                if let Some(function) = tc.get("function") {
                    let mut func_delta = serde_json::json!({});
                    if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                        func_delta["name"] = Value::String(name.to_string());
                    }
                    if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                        func_delta["arguments"] = Value::String(args.to_string());
                    }
                    delta_obj["function"] = func_delta;
                }

                chunks.push(StreamChunk::ToolCall(serde_json::json!({
                    "delta": {
                        "tool_calls": [{
                            "index": tc_index,
                            "id": delta_obj.get("id"),
                            "function": delta_obj.get("function"),
                        }]
                    }
                })));
            }
        }

        chunks
    }
}

#[async_trait]
impl RawAdapter for OpenAiProtocol {
    fn build_request(
        &self,
        request: &ChatRequest,
        mode: CallMode,
    ) -> Result<RawRequest, LlmError> {
        let stream = mode == CallMode::Stream;
        let body = self.build_openai_request(request, stream)?;

        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", self.api_key),
        );

        // Auto-add /v1 if base_url doesn't end with it
        let base = self.base_url.trim_end_matches('/');
        let url = if base.ends_with("/v1") {
            format!("{}/chat/completions", base)
        } else {
            format!("{}/v1/chat/completions", base)
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
                url = %request.url,
                error_body = %body,
                request_body = %serde_json::to_string(&request.body).unwrap_or_default(),
                "OpenAI API error with full request context"
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
                let data = es_event.data.trim();

                if data == "[DONE]" {
                    return futures_util::stream::iter(vec![Ok(StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    })]);
                }

                match serde_json::from_str::<Value>(data) {
                    Ok(json) => {
                        // Check for error object in stream (P1-5 fix)
                        if let Some(error) = json.get("error") {
                            let msg = error
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown error");
                            let err_type = error
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("error");
                            let code = error
                                .get("code")
                                .and_then(|c| c.as_str())
                                .unwrap_or("");
                            tracing::error!(
                                error_type = %err_type,
                                code = %code,
                                message = %msg,
                                "OpenAI stream error event"
                            );
                            return futures_util::stream::iter(vec![Err(LlmError::llm(
                                format!("{err_type}: {msg}")
                            ))]);
                        }

                        let mut chunks = Vec::new();

                        if let Some(usage) = json.get("usage") {
                            chunks.push(StreamChunk::Usage(UsageInfo {
                                prompt_tokens: usage
                                    .get("prompt_tokens")
                                    .and_then(|t| t.as_u64())
                                    .map(|n| n as u32),
                                completion_tokens: usage
                                    .get("completion_tokens")
                                    .and_then(|t| t.as_u64())
                                    .map(|n| n as u32),
                                total_tokens: usage
                                    .get("total_tokens")
                                    .and_then(|t| t.as_u64())
                                    .map(|n| n as u32),
                            }));
                        }

                        if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                let index = choice
                                    .get("index")
                                    .and_then(|i| i.as_u64())
                                    .unwrap_or(0) as usize;

                                if let Some(delta) = choice.get("delta") {
                                    chunks.extend(Self::convert_delta(delta, index));
                                }

                                if let Some(reason) =
                                    choice.get("finish_reason").and_then(|r| r.as_str())
                                {
                                    if reason == "length" || reason == "max_tokens" {
                                        tracing::warn!(
                                            finish_reason = reason,
                                            "stream truncated by token limit — tool call arguments may be incomplete"
                                        );
                                    }
                                    chunks.push(StreamChunk::Stop {
                                        finish_reason: Some(reason.to_string()),
                                    });
                                }
                            }
                        }

                        futures_util::stream::iter(chunks.into_iter().map(Ok).collect::<Vec<_>>())
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse SSE event: {e}, data: {data}");
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
        Self::parse_openai_response(body)
    }

    fn capabilities(&self) -> Capabilities {
        self.profile
            .as_ref()
            .map(|p| p.capabilities.clone())
            .unwrap_or_else(|| Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: true,
                supports_thinking: false,
                max_context_tokens: Some(128_000),
                max_output_tokens: Some(self.max_tokens),
            })
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: self
                .profile
                .as_ref()
                .map(|p| p.provider_name.to_string())
                .unwrap_or_else(|| "openai".to_string()),
            model: self.model.clone(),
            version: None,
        }
    }
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    use super::*;

    pub fn parse_openai_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        OpenAiProtocol::parse_openai_response(body)
    }

    pub fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        OpenAiProtocol::convert_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_trait::{Protocol, ReasoningConfig, ReasoningEffort, ReasoningMode};

    fn make_protocol() -> OpenAiProtocol {
        OpenAiProtocol::new("sk-test", "gpt-4o", None)
    }

    fn make_protocol_with_profile(profile: ModelProfile) -> OpenAiProtocol {
        OpenAiProtocol::new("sk-test", "gpt-4o", None).with_model_profile(profile)
    }

    fn mimo_openai_profile() -> ModelProfile {
        ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "mimo",
            capabilities: Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_thinking: true,
                max_context_tokens: Some(128_000),
                max_output_tokens: Some(8_192),
                ..Default::default()
            },
            reasoning_mode: ReasoningMode::None,
            supported_extra_params: &[],
        }
    }

    fn deepseek_profile() -> ModelProfile {
        ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "deepseek",
            capabilities: Capabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_thinking: true,
                max_context_tokens: Some(64_000),
                max_output_tokens: Some(8_192),
                ..Default::default()
            },
            reasoning_mode: ReasoningMode::Effort,
            supported_extra_params: &["reasoning_effort"],
        }
    }

    // ── MiMo: no reasoning_effort sent ──

    #[test]
    fn mimo_no_reasoning_effort() {
        let proto = make_protocol_with_profile(mimo_openai_profile());
        let reasoning = ReasoningConfig {
            effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        // MiMo's reasoning_mode is None, so reasoning_effort should NOT be sent
        assert!(
            raw.body.get("reasoning_effort").is_none(),
            "MiMo should not send reasoning_effort"
        );
    }

    // ── DeepSeek: reasoning_effort IS sent ──

    #[test]
    fn deepseek_sends_reasoning_effort() {
        let proto = make_protocol_with_profile(deepseek_profile());
        let reasoning = ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["reasoning_effort"], "high");
    }

    // ── Non-streaming reasoning_content extraction ──

    #[test]
    fn parse_response_with_reasoning_content() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        });
        let resp =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
                .unwrap();
        assert_eq!(resp.content, "The answer is 42.");
        assert_eq!(
            resp.reasoning_content.as_deref(),
            Some("Let me think step by step...")
        );
    }

    #[test]
    fn parse_response_without_reasoning_content() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
                .unwrap();
        assert!(resp.reasoning_content.is_none());
    }

    // ── convert_messages ──

    #[test]
    fn convert_messages_basic() {
        let msgs = vec![ChatMessage::user("hi")];
        let out = OpenAiProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"], "hi");
    }

    #[test]
    fn convert_messages_with_system() {
        let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let out = OpenAiProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn convert_messages_consecutive_assistant_must_merge() {
        // Two assistant messages: one with text, one with tool_calls.
        // P2-20: These should NOT be merged (tool_calls needs null content).
        let msgs = vec![
            ChatMessage::assistant("Let me check."),
            ChatMessage::assistant_tool_call("call_1", "search", r#"{"q":"test"}"#),
            ChatMessage::tool("call_1", "result"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        // Should be 3 messages: assistant(text), assistant(tool_calls), tool
        // because text+tool_calls should not be merged
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[0]["content"], "Let me check.");
        assert_eq!(out[1]["role"], "assistant");
        assert!(out[1].get("tool_calls").is_some());
        assert_eq!(out[2]["role"], "tool");
    }

    #[test]
    fn convert_messages_consecutive_user_must_merge() {
        let msgs = vec![
            ChatMessage::user("first question"),
            ChatMessage::user("follow up"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
    }

    #[test]
    fn convert_messages_consecutive_user_with_images_preserves_images() {
        // P0-3 FIXED: Consecutive user messages with images now merge as arrays.
        use llm_trait::ImageAttachment;
        let msgs = vec![
            ChatMessage::user_with_images(
                "describe this image",
                vec![ImageAttachment::Url {
                    url: "https://example.com/cat.jpg".to_string(),
                    detail: None,
                }],
            ),
            ChatMessage::user("what color is it?"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        let merged = &out[0];
        assert_eq!(merged["role"], "user");

        // Content should be an array (merged parts)
        let content = merged.get("content").unwrap();
        assert!(content.is_array(), "Content should be an array after merge");
        let parts = content.as_array().unwrap();

        // Should have: text("describe this image"), image_url, text("what color is it?")
        assert_eq!(parts.len(), 3, "Expected 3 parts: text, image, text");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe this image");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.jpg");
        assert_eq!(parts[2]["type"], "text");
        assert_eq!(parts[2]["text"], "what color is it?");
    }

    #[test]
    fn convert_messages_custom_filtered_out() {
        // P2-19: ChatMessage::Custom should be filtered out, not forwarded to LLM.
        let msgs = vec![
            ChatMessage::Custom {
                role: "metadata".to_string(),
                data: serde_json::json!({"secret": "should_not_leak"}),
            },
            ChatMessage::user("hello"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        // Custom message should be completely dropped
        assert_eq!(out.len(), 1, "P2-19: Custom message should be filtered out");
        let content = out[0]["content"].as_str().unwrap();
        assert!(
            !content.contains("should_not_leak"),
            "P2-19: Custom message data should not appear in output, got: {}",
            content
        );
    }

    #[test]
    fn convert_messages_assistant_tool_call_not_merged_with_text() {
        // P2-20: Merging an assistant+tool_calls message with an assistant+text message
        // produces an invalid request. They should not be merged.
        let msgs = vec![
            ChatMessage::assistant_tool_call("call_1", "shell", r#"{"cmd":"ls"}"#),
            ChatMessage::assistant("Here are the results: ..."),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        // Should be 2 separate messages, not merged
        assert_eq!(
            out.len(),
            2,
            "P2-20: assistant+tool_calls and assistant+text should not be merged"
        );
        // First should have tool_calls, no text content
        assert!(out[0].get("tool_calls").is_some());
        // Second should have text, no tool_calls
        assert_eq!(out[1]["content"], "Here are the results: ...");
    }

    // ── build_request ──

    #[test]
    fn build_request_basic() {
        let proto = make_protocol();
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Stream).unwrap();
        assert!(raw.url.contains("/chat/completions"));
        assert!(raw.stream);
        assert_eq!(raw.method, HttpMethod::Post);
        assert_eq!(raw.body["model"], "gpt-4o");
    }

    #[test]
    fn build_request_url_with_trailing_slash() {
        // Trailing slash should not cause double-slash in URL
        let proto = OpenAiProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1/"));
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert!(
            !raw.url.contains("//v1"),
            "URL should not have double-slash, got: {}",
            raw.url
        );
        assert!(
            raw.url.ends_with("/v1/chat/completions"),
            "URL should end with /v1/chat/completions, got: {}",
            raw.url
        );
    }

    #[test]
    fn build_request_stream_includes_usage_option() {
        // P2-18: OpenAI streaming should include stream_options.include_usage
        // so that usage data is returned in the stream.
        let proto = make_protocol();
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let raw = proto.build_request(&req, CallMode::Stream).unwrap();
        assert!(
            raw.body["stream_options"]["include_usage"].as_bool() == Some(true),
            "P2-18: stream requests should include stream_options.include_usage=true"
        );
    }

    // ── convert_delta ──

    #[test]
    fn convert_delta_text() {
        let delta = serde_json::json!({"content": "hello"});
        let chunks = OpenAiProtocol::convert_delta(&delta, 0);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Text(t) if t == "hello"));
    }

    #[test]
    fn convert_delta_reasoning_content() {
        let delta = serde_json::json!({"reasoning_content": "thinking..."});
        let chunks = OpenAiProtocol::convert_delta(&delta, 0);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::Thought(t) if t == "thinking..."));
    }

    // ── info ──

    #[test]
    fn info_with_profile() {
        let proto = make_protocol_with_profile(deepseek_profile());
        let info = proto.info();
        assert_eq!(info.name, "deepseek");
        assert_eq!(info.model, "gpt-4o");
    }

    #[test]
    fn info_without_profile() {
        let proto = make_protocol();
        let info = proto.info();
        assert_eq!(info.name, "openai");
    }

    #[test]
    fn parse_response_content_filter_finish_reason() {
        // P2-17: content_filter should map to FinishReason::ContentFilter, not Other
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": "content_filter"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
                .unwrap();
        assert_eq!(
            resp.finish_reason,
            FinishReason::ContentFilter,
            "P2-17: content_filter should map to ContentFilter, not Other"
        );
    }

    // ── parse_openai_response: error paths ──

    #[test]
    fn parse_response_invalid_json() {
        let result = OpenAiProtocol::parse_openai_response(b"not json");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse"));
    }

    #[test]
    fn parse_response_error_object() {
        let body = serde_json::json!({
            "error": {"message": "rate limited", "type": "rate_error"}
        });
        let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rate_error: rate limited"));
    }

    #[test]
    fn parse_response_error_missing_fields() {
        let body = serde_json::json!({"error": {}});
        let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("api_error: unknown error"));
    }

    #[test]
    fn parse_response_no_choices() {
        let body = serde_json::json!({"usage": {}});
        let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No choices"));
    }

    #[test]
    fn parse_response_no_message() {
        let body = serde_json::json!({"choices": [{}]});
        let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No message"));
    }

    // ── parse_openai_response: finish_reason variants ──

    #[test]
    fn parse_response_finish_reason_length() {
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length);
    }

    #[test]
    fn parse_response_finish_reason_unknown() {
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "something_new"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Other("something_new".into()));
    }

    #[test]
    fn parse_response_no_finish_reason_defaults_to_stop() {
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    // ── parse_openai_response: missing usage/tool_calls ──

    #[test]
    fn parse_response_no_usage() {
        let body = serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(resp.usage.prompt_tokens, None);
    }

    #[test]
    fn parse_response_tool_calls_missing_fields() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "fn", "arguments": "{}"}},
                        {"function": {"name": "fn2"}},
                        {"id": "t3"},
                        {}
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        // filter_map skips entries missing id or function or name
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "t1");
    }

    #[test]
    fn parse_response_reasoning_content_empty() {
        let body = serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "answer", "reasoning_content": ""},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
        // Empty reasoning_content is filtered out
        assert_eq!(resp.reasoning_content, None);
    }

    // ── capabilities ──

    #[test]
    fn capabilities_with_profile() {
        let profile = ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "openai",
            capabilities: Capabilities {
                supports_vision: false,
                supports_thinking: true,
                ..Default::default()
            },
            reasoning_mode: ReasoningMode::Effort,
            supported_extra_params: &[],
        };
        let proto = make_protocol().with_model_profile(profile);
        let caps = proto.capabilities();
        assert!(!caps.supports_vision);
        assert!(caps.supports_thinking);
    }

    #[test]
    fn capabilities_without_profile() {
        let proto = make_protocol();
        let caps = proto.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_vision);
        assert!(!caps.supports_thinking);
    }

    // ── from_config ──

    #[test]
    fn from_config_with_string_max_tokens() {
        let mut options = std::collections::HashMap::new();
        options.insert("max_tokens".into(), serde_json::json!("8192"));
        let config = LlmConfig {
            api_key: "sk-test".into(),
            model: "gpt-4".into(),
            base_url: "https://api.openai.com/v1".into(),
            options,
            ..Default::default()
        };
        let proto = OpenAiProtocol::from_config(&config);
        assert_eq!(proto.max_tokens, 8192);
    }

    #[test]
    fn from_config_default_max_tokens() {
        let config = LlmConfig {
            api_key: "sk-test".into(),
            model: "gpt-4".into(),
            base_url: "https://api.openai.com/v1".into(),
            ..Default::default()
        };
        let proto = OpenAiProtocol::from_config(&config);
        assert_eq!(proto.max_tokens, 16384);
    }

    // ── reasoning effort arms ──

    #[test]
    fn build_request_reasoning_low() {
        let profile = ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "openai",
            capabilities: Capabilities::default(),
            reasoning_mode: ReasoningMode::Effort,
            supported_extra_params: &[],
        };
        let proto = make_protocol().with_model_profile(profile);
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_reasoning(ReasoningConfig { effort: Some(ReasoningEffort::Low), ..Default::default() });
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["reasoning_effort"], "low");
    }

    #[test]
    fn build_request_reasoning_medium() {
        let profile = ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "openai",
            capabilities: Capabilities::default(),
            reasoning_mode: ReasoningMode::Effort,
            supported_extra_params: &[],
        };
        let proto = make_protocol().with_model_profile(profile);
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_reasoning(ReasoningConfig { effort: Some(ReasoningEffort::Medium), ..Default::default() });
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["reasoning_effort"], "medium");
    }

    #[test]
    fn build_request_reasoning_xhigh() {
        let profile = ModelProfile {
            protocol: Protocol::OpenAi,
            provider_name: "openai",
            capabilities: Capabilities::default(),
            reasoning_mode: ReasoningMode::Effort,
            supported_extra_params: &[],
        };
        let proto = make_protocol().with_model_profile(profile);
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_reasoning(ReasoningConfig { effort: Some(ReasoningEffort::XHigh), ..Default::default() });
        let raw = proto.build_request(&req, CallMode::Once).unwrap();
        assert_eq!(raw.body["reasoning_effort"], "high");
    }

    // ── convert_messages: images ──

    #[test]
    fn convert_messages_base64_image() {
        use llm_trait::ImageAttachment;
        let msgs = vec![ChatMessage::user_with_images(
            "describe",
            vec![ImageAttachment::Base64 {
                data: "abc123".into(),
                media_type: Some("image/png".into()),
                detail: None,
            }],
        )];
        let out = OpenAiProtocol::convert_messages(&msgs);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert!(content[1]["image_url"]["url"].as_str().unwrap().contains("data:image/png;base64,abc123"));
    }

    #[test]
    fn convert_messages_url_image_with_detail() {
        use llm_trait::ImageAttachment;
        let msgs = vec![ChatMessage::user_with_images(
            "describe",
            vec![ImageAttachment::Url {
                url: "https://example.com/img.jpg".into(),
                detail: Some(llm_trait::ImageDetail::High),
            }],
        )];
        let out = OpenAiProtocol::convert_messages(&msgs);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content[1]["image_url"]["detail"], "high");
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
                let out = OpenAiProtocol::convert_messages(&messages);
                // Merging can only reduce or maintain count; Custom messages are filtered
                let non_custom_count = messages.iter().filter(|m| !matches!(m, ChatMessage::Custom { .. })).count();
                assert!(out.len() <= non_custom_count,
                    "output {} > non-custom input {}", out.len(), non_custom_count);
            }

            #[test]
            fn convert_messages_output_roles_are_valid(
                messages in prop::collection::vec(arb_chat_message(), 0..20)
            ) {
                let out = OpenAiProtocol::convert_messages(&messages);
                let valid_roles = ["system", "user", "assistant", "tool"];
                for msg in &out {
                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    assert!(valid_roles.contains(&role),
                        "invalid role: {:?}, full message: {:?}", role, msg);
                }
            }

            #[test]
            fn convert_messages_single_user_preserved(
                content in "[a-z ]{1,50}"
            ) {
                let msgs = vec![ChatMessage::user(&content)];
                let out = OpenAiProtocol::convert_messages(&msgs);
                assert_eq!(out.len(), 1);
                assert_eq!(out[0]["role"], "user");
                assert_eq!(out[0]["content"], content);
            }
        }
    }
}
