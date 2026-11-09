//! Anthropic protocol implementation.
//!
//! Implements `RawAdapter` for the Anthropic Messages API.
//! Uses `eventsource-stream` for SSE parsing,
//! with `anthropic-rs-api` types for request/response structures.

mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

use llm_trait::{
    CallMode, Capabilities, ChatRequest, ChatResponse, ChatStream, HttpClient, HttpMethod,
    LlmConfig, LlmError, ProviderInfo, RawAdapter, RawRequest,
};

use super::types::MessagesStreamEvent;

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
            base_url: base_url.unwrap_or("https://api.anthropic.com").to_string(),
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
impl RawAdapter for AnthropicProtocol {
    fn build_request(&self, request: &ChatRequest, mode: CallMode) -> Result<RawRequest, LlmError> {
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
                let data = es_event.data.trim();
                // Non-JSON sentinels / keep-alives are not event payloads — just skip:
                // - `[DONE]`: OpenAI-style stream terminator (some relays/proxies send it
                //   again after message_stop)
                // - `: ...`: upstream comment-style heartbeat (e.g. MiMo sends `data: : PROCESSING`)
                // Anthropic's end-of-stream signal comes from message_stop
                if data == "[DONE]" || data.starts_with(':') {
                    return futures_util::stream::iter(vec![]);
                }
                match serde_json::from_str::<MessagesStreamEvent>(data) {
                    Ok(stream_event) => match Self::convert_event(stream_event) {
                        Ok(chunks) => futures_util::stream::iter(
                            chunks.into_iter().map(Ok).collect::<Vec<_>>(),
                        ),
                        Err(e) => futures_util::stream::iter(vec![Err(e)]),
                    },
                    Err(e) => {
                        tracing::warn!("Failed to parse SSE event: {e}, data: {}", es_event.data);
                        // Propagate parse error instead of swallowing
                        futures_util::stream::iter(vec![Err(LlmError::stream(format!(
                            "Failed to parse SSE event: {e}"
                        )))])
                    }
                }
            }
            Err(e) => {
                futures_util::stream::iter(vec![Err(LlmError::stream(format!("SSE error: {e}")))])
            }
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
    use llm_trait::ChatMessage;
    use serde_json::Value;

    pub fn parse_anthropic_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        AnthropicProtocol::parse_anthropic_response(body)
    }

    pub fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        AnthropicProtocol::convert_messages(messages)
    }
}
