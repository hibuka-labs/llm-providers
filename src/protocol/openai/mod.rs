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
    Capabilities, CallMode, ChatRequest, ChatResponse, ChatStream, FinishReason, HttpMethod,
    ImageAttachment, ImageDetail, LlmBackend, LlmConfig, LlmError, ProviderInfo, RawAdapter,
    RawRequest, StreamChunk, ToolCall, UsageInfo, ChatMessage,
};

/// OpenAI Chat Completions protocol implementation.
///
/// Handles request building, SSE stream parsing, and response conversion
/// for the OpenAI Chat Completions API (`/v1/chat/completions`).
pub struct OpenAiProtocol {
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
}

impl OpenAiProtocol {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: base_url
                .unwrap_or("https://api.openai.com/v1")
                .to_string(),
            max_tokens: 4096,
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        let max_tokens = config
            .options
            .get("max_tokens")
            .and_then(|v| v.as_str().and_then(|s| s.parse::<u32>().ok()).or_else(|| v.as_u64().map(|n| n as u32)))
            .unwrap_or(4096);

        Self::new(&config.api_key, &config.model, Some(&config.base_url))
            .with_max_tokens(max_tokens)
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
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

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| LlmError::llm(format!("Failed to serialize tools: {e}")))?;
        }

        // Handle reasoning/thinking (extended thinking via reasoning_effort)
        if let Some(ref rc) = request.reasoning {
            if let Some(ref effort) = rc.effort {
                let effort_str = match effort {
                    llm_trait::ReasoningEffort::Low => Some("low"),
                    llm_trait::ReasoningEffort::Medium => Some("medium"),
                    llm_trait::ReasoningEffort::High => Some("high"),
                    _ => None,
                };
                if let Some(effort_str) = effort_str {
                    body["reasoning_effort"] = Value::String(effort_str.to_string());
                }
            }
        }

        Ok(body)
    }

    /// Convert ChatMessages to OpenAI message format.
    ///
    /// Key differences from Anthropic:
    /// - System messages go into the messages array (not a top-level field)
    /// - User content can be a string or array of content parts
    /// - Tool results use role "tool" (not wrapped in user message)
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
                        // Multi-part content with images
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
                } => {
                    result.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content
                    }));
                }
                ChatMessage::Custom { role: _, data } => {
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": data.to_string()
                    }));
                }
            }
        }

        // Merge consecutive same-role messages to satisfy OpenAI API requirements.
        // OpenAI rejects requests with consecutive same-role messages (400 error).
        // This can happen when agent_base saves text and tool_calls as separate messages.
        let mut merged: Vec<Value> = Vec::with_capacity(result.len());
        for msg in result {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            let can_merge = if let Some(prev) = merged.last() {
                let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == prev_role && role != "tool" {
                    // Merge consecutive same-role messages (except tool messages)
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if can_merge {
                if let Some(prev) = merged.last_mut() {
                    // Merge assistant messages: combine content and tool_calls
                    if role == "assistant" {
                        // Combine content
                        let prev_content = prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let new_content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
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

                        // Combine tool_calls
                        if let Some(new_tc) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                            if let Some(prev_tc) = prev.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                                prev_tc.extend(new_tc.iter().cloned());
                            } else {
                                prev["tool_calls"] = Value::Array(new_tc.clone());
                            }
                        }
                        continue;
                    }
                    // Merge user messages: concatenate content
                    if role == "user" {
                        let prev_content = prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let new_content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let combined = if prev_content.is_empty() {
                            new_content.to_string()
                        } else if new_content.is_empty() {
                            prev_content.to_string()
                        } else {
                            format!("{}\n{}", prev_content, new_content)
                        };
                        prev["content"] = Value::String(combined);
                        continue;
                    }
                }
            }

            merged.push(msg);
        }

        merged
    }

    /// Convert tools to OpenAI format.
    ///
    /// OpenAI tools format: `{ "type": "function", "function": { "name": ..., "parameters": ... } }`
    /// If tools are already in this format, they pass through.
    fn convert_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .map(|tool| {
                // If already in OpenAI format (has "type" and "function"), pass through
                if tool.get("type").is_some() && tool.get("function").is_some() {
                    return tool.clone();
                }
                // Otherwise assume it's already in the right format or wrap it
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

        // Extract the first choice
        let choice = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| LlmError::llm("No choices in response"))?;

        let message = choice
            .get("message")
            .ok_or_else(|| LlmError::llm("No message in choice"))?;

        // Extract content
        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Extract tool calls
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

        // Extract usage
        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u.get("prompt_tokens").and_then(|t| t.as_u64()).map(|n| n as u32),
                completion_tokens: u
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                total_tokens: u.get("total_tokens").and_then(|t| t.as_u64()).map(|n| n as u32),
            })
            .unwrap_or_default();

        // Extract finish reason
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::Other("content_filter".to_string()),
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

    /// Convert a single OpenAI SSE delta to StreamChunk(s).
    ///
    /// Handles incremental text and tool call assembly.
    /// Tool calls arrive incrementally: first chunk has `id` + `name`,
    /// subsequent chunks have only `arguments` fragments.
    fn convert_delta(delta: &Value, index: usize) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();

        // Text content delta
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                chunks.push(StreamChunk::Text(content.to_string()));
            }
        }

        // Reasoning / thinking delta (for o1/o3-style models)
        if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
            if !reasoning.is_empty() {
                chunks.push(StreamChunk::Thought(reasoning.to_string()));
            }
        }

        // Tool call deltas
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let tc_index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(index as u64) as usize;
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

        Ok(RawRequest {
            url: format!("{}/chat/completions", self.base_url),
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
            tracing::error!(
                status = status.as_u16(),
                url = %request.url,
                error_body = %body,
                request_body = %serde_json::to_string(&request.body).unwrap_or_default(),
                "OpenAI API error with full request context"
            );
            return Err(LlmError::api(status.as_u16(), body));
        }

        // Parse SSE stream
        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        let chunk_stream = event_stream.flat_map(|event| match event {
            Ok(es_event) => {
                let data = es_event.data.trim();

                // Check for stream termination
                if data == "[DONE]" {
                    return futures_util::stream::iter(vec![Ok(StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    })]);
                }

                match serde_json::from_str::<Value>(data) {
                    Ok(json) => {
                        // Extract usage from the final chunk (if present)
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

                        // Process choices
                        if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                let index = choice
                                    .get("index")
                                    .and_then(|i| i.as_u64())
                                    .unwrap_or(0) as usize;

                                if let Some(delta) = choice.get("delta") {
                                    chunks.extend(Self::convert_delta(delta, index));
                                }

                                // Check for finish_reason
                                if let Some(reason) =
                                    choice.get("finish_reason").and_then(|r| r.as_str())
                                {
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
        Self::parse_openai_response(body)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: false, // Most OpenAI models don't support thinking
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(self.max_tokens),
        }
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "openai".to_string(),
            model: self.model.clone(),
            backend: LlmBackend::OpenAi,
            version: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_protocol() -> OpenAiProtocol {
        OpenAiProtocol::new("sk-test", "gpt-4o", None)
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
        assert_eq!(out[0]["content"], "sys");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn convert_messages_with_images() {
        let msgs = vec![ChatMessage::user_with_images(
            "pic",
            vec![ImageAttachment::Url {
                url: "http://example.com/img.png".into(),
                detail: None,
            }],
        )];
        let out = OpenAiProtocol::convert_messages(&msgs);
        assert_eq!(out.len(), 1);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2); // text + image
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn convert_messages_with_tool_calls() {
        let msgs = vec![
            ChatMessage::assistant("Let me check."),
            ChatMessage::assistant_tool_call("call_1", "search", r#"{"q":"test"}"#),
            ChatMessage::tool("call_1", "result"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);
        // Assistant text + assistant tool_call are separate messages
        assert!(out.iter().any(|m| m["role"] == "assistant"));
        assert!(out.iter().any(|m| m["role"] == "tool"));
    }

    /// Regression test: consecutive assistant messages must NOT appear in output.
    ///
    /// When the agent has both text and tool_call, agent_base saves them as TWO
    /// separate assistant messages (push_assistant_with_reasoning + push_assistant_tool_calls).
    /// OpenAI API rejects consecutive same-role messages with 400.
    /// This test verifies that convert_messages merges them.
    #[test]
    fn convert_messages_consecutive_assistant_must_merge() {
        let msgs = vec![
            // Simulates: assistant returned text "Let me check" + tool_call "search"
            // agent_base saves as two messages:
            ChatMessage::assistant("Let me check."),
            ChatMessage::assistant_tool_call("call_1", "search", r#"{"q":"test"}"#),
            ChatMessage::tool("call_1", "result"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);

        // Check: no two consecutive same-role messages
        for i in 1..out.len() {
            let prev_role = out[i - 1]["role"].as_str().unwrap();
            let curr_role = out[i]["role"].as_str().unwrap();
            assert_ne!(
                prev_role, curr_role,
                "consecutive same-role messages at index {}: role='{}'",
                i - 1, prev_role
            );
        }
    }

    /// Regression test: consecutive user messages must NOT appear in output.
    ///
    /// When compression injects a follow-up user message right after a tool_result,
    /// the message history can have consecutive user messages.
    /// OpenAI API rejects this with 400.
    #[test]
    fn convert_messages_consecutive_user_must_merge() {
        let msgs = vec![
            ChatMessage::user("first question"),
            ChatMessage::user("follow up"),
        ];
        let out = OpenAiProtocol::convert_messages(&msgs);

        // Should be merged into one user message
        assert_eq!(out.len(), 1, "consecutive user messages should be merged");
        assert_eq!(out[0]["role"], "user");
    }

    // ── convert_tools ──

    #[test]
    fn convert_tools_passthrough() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo back",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let out = OpenAiProtocol::convert_tools(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["function"]["name"], "echo");
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
        assert!(raw.headers.contains_key("Authorization"));
        assert!(raw.headers["Authorization"].starts_with("Bearer "));
        assert_eq!(raw.body["model"], "gpt-4o");
        assert_eq!(raw.body["stream"], true);
        assert_eq!(raw.body["max_tokens"], 4096);
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
        let messages = raw.body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful");
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
        let tools = raw.body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "echo");
    }

    // ── parse_openai_response ──

    #[test]
    fn parse_response_text_only() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });
        let resp =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
                .unwrap();
        assert_eq!(resp.content, "Hello!");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.prompt_tokens, Some(10));
        assert_eq!(resp.usage.completion_tokens, Some(5));
        assert_eq!(resp.usage.total_tokens, Some(15));
    }

    #[test]
    fn parse_response_with_tool_calls() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"q\":\"test\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });
        let resp =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
                .unwrap();
        assert_eq!(resp.content, "");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "search");
        assert_eq!(resp.tool_calls[0].arguments, "{\"q\":\"test\"}");
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn parse_response_api_error() {
        let body = serde_json::json!({
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid API key"
            }
        });
        let result =
            OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("Invalid API key"));
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
    fn convert_delta_tool_call_start() {
        let delta = serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "search", "arguments": ""}
            }]
        });
        let chunks = OpenAiProtocol::convert_delta(&delta, 0);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall(_)));
    }

    #[test]
    fn convert_delta_tool_call_arguments() {
        let delta = serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": "{\"q\":\""}
            }]
        });
        let chunks = OpenAiProtocol::convert_delta(&delta, 0);
        assert_eq!(chunks.len(), 1);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall(_)));
    }

    #[test]
    fn convert_delta_empty_content() {
        let delta = serde_json::json!({"content": null});
        let chunks = OpenAiProtocol::convert_delta(&delta, 0);
        assert!(chunks.is_empty());
    }

    // ── capabilities / info ──

    #[test]
    fn capabilities_correct() {
        let proto = make_protocol();
        let caps = proto.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_vision);
        assert!(!caps.supports_thinking);
        assert_eq!(caps.max_context_tokens, Some(128_000));
    }

    #[test]
    fn info_correct() {
        let proto = make_protocol();
        let info = proto.info();
        assert_eq!(info.name, "openai");
        assert_eq!(info.model, "gpt-4o");
        assert_eq!(info.backend, LlmBackend::OpenAi);
    }

    #[test]
    fn from_config() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "gpt-4o".to_string(),
            base_url: "https://custom.api.com/v1".to_string(),
            options: Default::default(),
        };
        let proto = OpenAiProtocol::from_config(&config);
        assert_eq!(proto.model, "gpt-4o");
        assert_eq!(proto.base_url, "https://custom.api.com/v1");
    }
}
