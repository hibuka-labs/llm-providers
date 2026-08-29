//! Raw adapter trait for low-level provider implementations.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::capabilities::{Capabilities, ProviderInfo};
use super::error::LlmError;
use super::http_client::HttpClient;
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
///
/// `headers` normally carry the API key, so the `Debug` impl redacts values for
/// authentication-related header names.
#[derive(Clone)]
pub struct RawRequest {
    pub url: String,
    pub method: HttpMethod,
    pub headers: HashMap<String, String>,
    pub body: Value,
    pub stream: bool,
}

/// Whether a header name looks like it carries a credential.
fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("authorization")
        || name.contains("api-key")
        || name.contains("apikey")
        || name.contains("x-api-key")
        || name.contains("token")
}

impl std::fmt::Debug for RawRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let headers: HashMap<&str, &str> = self
            .headers
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str(),
                    if is_sensitive_header(k) {
                        "***"
                    } else {
                        v.as_str()
                    },
                )
            })
            .collect();
        f.debug_struct("RawRequest")
            .field("url", &self.url)
            .field("method", &self.method)
            .field("headers", &headers)
            .field("body", &self.body)
            .field("stream", &self.stream)
            .finish()
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    fn request_with(headers: &[(&str, &str)]) -> RawRequest {
        RawRequest {
            url: "https://api.example.com/v1/messages".to_string(),
            method: HttpMethod::Post,
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body: serde_json::json!({"model": "m"}),
            stream: true,
        }
    }

    #[test]
    fn credential_headers_are_redacted() {
        let secret = "sk-super-secret-value";
        for name in [
            "authorization",
            "Authorization",
            "x-api-key",
            "X-API-KEY",
            "api-key",
            "apikey",
            "x-goog-api-key",
            "x-session-token",
            "bearer-token",
        ] {
            let rendered = format!("{:?}", request_with(&[(name, secret)]));
            assert!(
                !rendered.contains(secret),
                "header '{name}' leaked its value: {rendered}"
            );
            assert!(
                rendered.contains("***"),
                "header '{name}' should be redacted: {rendered}"
            );
        }
    }

    #[test]
    fn non_credential_headers_stay_visible() {
        // Content type and version headers are not secrets; hiding them would
        // make debugging protocol mismatches needlessly hard.
        let rendered = format!(
            "{:?}",
            request_with(&[
                ("content-type", "application/json"),
                ("anthropic-version", "2023-06-01"),
            ])
        );
        assert!(rendered.contains("application/json"));
        assert!(rendered.contains("2023-06-01"));
        assert!(!rendered.contains("***"));
    }

    #[test]
    fn other_fields_remain_visible() {
        let rendered = format!("{:?}", request_with(&[("x-api-key", "secret")]));
        assert!(rendered.contains("https://api.example.com/v1/messages"));
        assert!(rendered.contains("Post"));
        assert!(rendered.contains("stream: true"));
        assert!(rendered.contains("model"));
    }

    #[test]
    fn is_sensitive_header_recognises_credential_names() {
        for name in [
            "authorization",
            "X-API-Key",
            "apikey",
            "refresh_token",
            "API-KEY",
        ] {
            assert!(is_sensitive_header(name), "{name} should be sensitive");
        }
        for name in ["content-type", "user-agent", "anthropic-version", "accept"] {
            assert!(!is_sensitive_header(name), "{name} should not be sensitive");
        }
    }
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
#[allow(dead_code)]
pub struct StreamState {
    pub data: HashMap<String, Value>,
}

/// Low-level adapter trait.
///
/// Implement this to get full `LlmProvider` functionality,
/// wrapped automatically by `GenericProvider`.
///
/// Implementors must provide:
/// - [`build_request`](RawAdapter::build_request) - Build HTTP request
/// - [`execute_stream`](RawAdapter::execute_stream) - Required to implement, but
///   `GenericProvider` does not call it; it is the hook for adapters that own the
///   whole send-and-parse flow
/// - [`parse_sse_stream`](RawAdapter::parse_sse_stream) - Parse SSE from a
///   pre-fetched response. **Override this for streaming to work**: the default
///   returns an error, and it is this method `GenericProvider` calls.
/// - [`capabilities`](RawAdapter::capabilities) - Declare capabilities
/// - [`info`](RawAdapter::info) - Provide info
///
/// Optional:
/// - [`parse_response`](RawAdapter::parse_response) - Parse non-streaming response
///   (default returns an error). `chat()` only falls back to streaming when
///   `supported_modes()` omits `CallMode::Once` — since that is not the default,
///   an adapter that does not implement `parse_response` must also narrow
///   `supported_modes`, or `chat()` will return the default error.
///
/// Note: the adapter is fully responsible for stream parsing, including SSE frame
/// parsing and incremental tool call assembly. GenericProvider does not get involved
/// in stream details.
#[async_trait]
pub trait RawAdapter: Send + Sync {
    /// Build HTTP request.
    fn build_request(&self, request: &ChatRequest, mode: CallMode) -> Result<RawRequest, LlmError>;

    /// Execute streaming request, fully parse SSE response.
    ///
    /// The implementor receives `&dyn HttpClient` and parses the stream protocol,
    /// returning `ChatStream`. This gives the adapter full control,
    /// suitable for different providers' SSE format differences.
    ///
    /// Typical implementation:
    /// 1. Send request via `client.send(&request)`
    /// 2. Check HTTP status code
    /// 3. Parse SSE stream (according to provider's protocol format)
    /// 4. Incremental tool call assembly (if needed)
    /// 5. Return ChatStream
    async fn execute_stream(
        &self,
        client: &dyn HttpClient,
        request: RawRequest,
    ) -> Result<ChatStream, LlmError>;

    /// Parse SSE stream from an already-received HTTP response.
    ///
    /// This method is called by GenericProvider after a successful HTTP response
    /// (with retry logic applied). The adapter only needs to parse the SSE stream,
    /// not send the HTTP request.
    ///
    /// Default implementation: returns an error. Adapters must override this —
    /// GenericProvider calls it rather than [`execute_stream`](Self::execute_stream)
    /// once a response is in hand, so an unimplemented override surfaces as a
    /// stream error rather than a silently re-sent request.
    async fn parse_sse_stream(
        &self,
        _client: &dyn HttpClient,
        _request: RawRequest,
        _response: super::http_client::HttpResponse,
    ) -> Result<ChatStream, LlmError> {
        Err(LlmError::llm("parse_sse_stream not implemented"))
    }

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
