//! OpenAI Chat Completions protocol implementation.
//!
//! Implements `RawAdapter` for the OpenAI Chat Completions API.
//! Also serves as the base protocol for OpenAI-compatible providers
//! (DeepSeek, Qwen, Azure, local models, etc.).

mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

use llm_trait::{
    CallMode, Capabilities, ChatRequest, ChatResponse, ChatStream, HttpClient, HttpMethod,
    LlmConfig, LlmError, ProviderInfo, RawAdapter, RawRequest, StreamChunk, UsageInfo,
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
            base_url: base_url.unwrap_or("https://api.openai.com/v1").to_string(),
            max_tokens: 16_384, // default 16K
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
            .unwrap_or(16_384); // default 16K

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
}

#[async_trait]
impl RawAdapter for OpenAiProtocol {
    fn build_request(&self, request: &ChatRequest, mode: CallMode) -> Result<RawRequest, LlmError> {
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

                // Upstream comment-style heartbeats (e.g. MiMo sends `data: : PROCESSING`)
                // are not event payloads — skip them
                if data.starts_with(':') {
                    return futures_util::stream::iter(vec![]);
                }

                match serde_json::from_str::<Value>(data) {
                    Ok(json) => {
                        // Check for error object in stream
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
                                reasoning_tokens: usage
                                    .get("completion_tokens_details")
                                    .and_then(|d| d.get("reasoning_tokens"))
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
                max_context_tokens: Some(1_000_000),
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
    use llm_trait::ChatMessage;

    pub fn parse_openai_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        OpenAiProtocol::parse_openai_response(body)
    }

    pub fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        OpenAiProtocol::convert_messages(messages)
    }
}
