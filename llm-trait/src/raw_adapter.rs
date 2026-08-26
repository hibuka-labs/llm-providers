//! Raw adapter trait for low-level provider implementations.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::capabilities::{Capabilities, ProviderInfo};
use super::error::LlmError;
use super::request::ChatRequest;
use super::response::{ChatResponse, ChatStream};

/// Call mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallMode {
    /// Streaming response
    Stream,
    /// Non-streaming response
    Once,
}

/// Raw HTTP request.
#[derive(Debug, Clone)]
pub struct RawRequest {
    pub url: String,
    pub method: HttpMethod,
    pub headers: HashMap<String, String>,
    pub body: Value,
    pub stream: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum HttpMethod {
    #[default]
    Post,
    Get,
    Put,
    Delete,
}

/// Stream parsing state for adapter-internal incremental assembly.
///
/// For example, Anthropic tool calls need to assemble arguments across
/// multiple SSE events. Adapters can use StreamState to buffer intermediate state.
#[derive(Debug, Default)]
pub struct StreamState {
    pub data: HashMap<String, Value>,
}

/// Low-level adapter trait.
///
/// Implement this to get full `LlmProvider` functionality,
/// wrapped automatically by `GenericProvider`.
///
/// Implementors need:
/// - [`build_request`](RawAdapter::build_request) - Build HTTP request
/// - [`execute_stream`](RawAdapter::execute_stream) - Execute streaming request and parse SSE
/// - [`parse_response`](RawAdapter::parse_response) - Parse non-streaming response (optional)
/// - [`capabilities`](RawAdapter::capabilities) - Declare capabilities
/// - [`info`](RawAdapter::info) - Provide info
///
/// Note: the adapter is fully responsible for stream parsing, including SSE frame
/// parsing and incremental tool call assembly. GenericProvider does not介入 stream details.
#[async_trait]
pub trait RawAdapter: Send + Sync {
    /// Build HTTP request.
    fn build_request(
        &self,
        request: &ChatRequest,
        mode: CallMode,
    ) -> Result<RawRequest, LlmError>;

    /// Execute streaming request, fully parse SSE response.
    ///
    /// The implementor receives `reqwest::Response` and parses the stream protocol,
    /// returning `ChatStream`. This gives the adapter full control,
    /// suitable for different providers' SSE format differences.
    ///
    /// Typical implementation:
    /// 1. Check HTTP status code
    /// 2. Parse SSE stream (according to provider's protocol format)
    /// 3. Incremental tool call assembly (if needed)
    /// 4. Return ChatStream
    async fn execute_stream(
        &self,
        client: &reqwest::Client,
        request: RawRequest,
    ) -> Result<ChatStream, LlmError>;

    /// Parse non-streaming response.
    ///
    /// Default: not supported. Override this if the adapter supports non-streaming mode.
    fn parse_response(&self, _body: &[u8]) -> Result<ChatResponse, LlmError> {
        Err(LlmError::llm("Non-streaming mode not supported"))
    }

    /// Get capabilities.
    fn capabilities(&self) -> Capabilities;

    /// Get info.
    fn info(&self) -> ProviderInfo;

    /// Supported call modes.
    fn supported_modes(&self) -> &[CallMode] {
        &[CallMode::Stream, CallMode::Once]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_mode_equality() {
        assert_eq!(CallMode::Stream, CallMode::Stream);
        assert_eq!(CallMode::Once, CallMode::Once);
        assert_ne!(CallMode::Stream, CallMode::Once);
    }

    #[test]
    fn http_method_default_is_post() {
        assert_eq!(HttpMethod::default(), HttpMethod::Post);
    }

    #[test]
    fn raw_request_clone() {
        let req = RawRequest {
            url: "https://api.example.com/v1/chat".to_string(),
            method: HttpMethod::Post,
            headers: HashMap::from([("Authorization".to_string(), "Bearer sk-xxx".to_string())]),
            body: serde_json::json!({"model": "test"}),
            stream: true,
        };
        let cloned = req.clone();
        assert_eq!(cloned.url, req.url);
        assert_eq!(cloned.method, req.method);
        assert_eq!(cloned.stream, req.stream);
    }

    #[test]
    fn stream_state_default() {
        let state = StreamState::default();
        assert!(state.data.is_empty());
    }
}
