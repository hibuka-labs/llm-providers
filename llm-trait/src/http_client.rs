//! HTTP client abstraction trait.
//!
//! Decouples `RawAdapter` from `reqwest::Client`, enabling mock testing
//! and alternative HTTP client implementations.

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;

use super::error::LlmError;
use super::raw_adapter::RawRequest;

/// HTTP client trait for sending requests.
///
/// Object-safe: `&dyn HttpClient` works.
/// GenericProvider owns a concrete implementation and passes `&dyn HttpClient`
/// to adapters via `execute_stream`.
#[async_trait]
pub trait HttpClient: Send + Sync {
    /// Send an HTTP request and return the response.
    async fn send(&self, request: &RawRequest) -> Result<HttpResponse, LlmError>;
}

/// Type-erased byte stream for SSE parsing.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, LlmError>> + Send>>;

/// HTTP response abstraction.
///
/// Minimal wrapper around an HTTP response, exposing only what adapters need.
/// No reqwest types are exposed in the public API.
pub struct HttpResponse {
    status: u16,
    body_stream: Option<ByteStream>,
    body_text: Option<String>,
}

impl HttpResponse {
    /// Create from parts (used by ReqwestHttpClient).
    pub fn new(status: u16, body_stream: ByteStream) -> Self {
        Self {
            status,
            body_stream: Some(body_stream),
            body_text: None,
        }
    }

    /// Create from a pre-read text body (for error responses).
    pub fn from_text(status: u16, body_text: String) -> Self {
        Self {
            status,
            body_stream: None,
            body_text: Some(body_text),
        }
    }

    /// HTTP status code.
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Whether the response is a success (2xx).
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Read the response body as text.
    pub async fn text(self) -> String {
        if let Some(text) = self.body_text {
            return text;
        }

        // Read from stream
        if let Some(mut stream) = self.body_stream {
            use futures_util::StreamExt;
            let mut body = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => body.push_str(&String::from_utf8_lossy(&bytes)),
                    Err(_) => break,
                }
            }
            body
        } else {
            String::new()
        }
    }

    /// Get a byte stream for SSE parsing.
    pub fn bytes_stream(self) -> ByteStream {
        self.body_stream
            .unwrap_or_else(|| Box::pin(futures_util::stream::empty()))
    }
}

/// Default HTTP client implementation using reqwest.
pub struct ReqwestHttpClient {
    client: reqwest::Client,
}

impl ReqwestHttpClient {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl HttpClient for ReqwestHttpClient {
    async fn send(&self, request: &RawRequest) -> Result<HttpResponse, LlmError> {
        use super::raw_adapter::HttpMethod;

        let mut builder = match request.method {
            HttpMethod::Post => self.client.post(&request.url),
            HttpMethod::Get => self.client.get(&request.url),
            HttpMethod::Put => self.client.put(&request.url),
            HttpMethod::Delete => self.client.delete(&request.url),
        };

        for (key, value) in &request.headers {
            builder = builder.header(key.as_str(), value.as_str());
        }

        builder = builder
            .header("Content-Type", "application/json")
            .json(&request.body);

        let response = builder.send().await.map_err(|e| {
            tracing::error!(error = %e, url = %request.url, "HTTP request failed");
            LlmError::llm(format!("HTTP request failed: {e}"))
        })?;

        let status = response.status().as_u16();

        // Convert reqwest byte stream to our type-erased stream
        let reqwest_stream = response.bytes_stream();
        let byte_stream = Box::pin(reqwest_stream.map(|r| {
            r.map_err(|e| LlmError::stream(format!("Stream read error: {e}")))
        }));

        Ok(HttpResponse::new(status, byte_stream))
    }
}
