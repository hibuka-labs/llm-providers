//! Generic provider implementation.
//!
//! `GenericProvider<A: RawAdapter>` wraps any `RawAdapter` and provides:
//! - HTTP client management (with optional external client injection)
//! - Error handling and retry logic (exponential backoff for 429/5xx)
//! - Automatic `LlmProvider` trait implementation

use std::time::Duration;

use async_trait::async_trait;

use llm_trait::{
    Capabilities, CallMode, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
    ProviderInfo, RawAdapter, RawRequest,
};

/// Provider configuration
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub retry_delay: Duration,
    /// Optional external HTTP client (for connection pool sharing)
    pub client: Option<reqwest::Client>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(120),
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            client: None,
        }
    }
}

/// Generic LLM provider built on top of any `RawAdapter`.
///
/// Handles HTTP client management, retry logic, and automatically
/// implements `LlmProvider` for any `A: RawAdapter`.
pub struct GenericProvider<A: RawAdapter> {
    adapter: A,
    client: reqwest::Client,
    config: ProviderConfig,
}

impl<A: RawAdapter> GenericProvider<A> {
    pub fn new(adapter: A) -> Self {
        Self::with_config(adapter, ProviderConfig::default())
    }

    pub fn with_config(adapter: A, config: ProviderConfig) -> Self {
        let client = config.client.clone().unwrap_or_else(|| {
            reqwest::Client::builder()
                .connect_timeout(config.connect_timeout)
                .read_timeout(config.request_timeout)
                .build()
                .expect("Failed to build HTTP client")
        });

        Self {
            adapter,
            client,
            config,
        }
    }

    /// Get a reference to the inner adapter.
    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Execute a non-streaming request.
    async fn execute_once(&self, request: RawRequest) -> Result<ChatResponse, LlmError> {
        let response = self.send_request(request).await?;
        let body = response.bytes().await
            .map_err(|e| LlmError::llm(format!("Failed to read response body: {e}")))?;
        self.adapter.parse_response(&body)
    }

    /// Execute a streaming request (delegated to adapter).
    async fn execute_stream(&self, request: RawRequest) -> Result<ChatStream, LlmError> {
        self.adapter.execute_stream(&self.client, request).await
    }

    /// Send HTTP request with retry logic.
    async fn send_request(&self, request: RawRequest) -> Result<reqwest::Response, LlmError> {
        let mut last_err = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.calculate_backoff(attempt);
                tokio::time::sleep(delay).await;
            }

            match self.do_send_request(&request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if !self.is_retryable(&e) || attempt == self.config.max_retries {
                        return Err(e);
                    }
                    tracing::warn!(attempt, error = %e, "Request failed, retrying");
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| LlmError::llm("Request failed after retries")))
    }

    fn is_retryable(&self, error: &LlmError) -> bool {
        match error {
            LlmError::LlmApi { status, .. } => {
                *status == 429 || *status >= 500
            }
            LlmError::Llm(_) | LlmError::Stream(_) => true,
            _ => false,
        }
    }

    fn calculate_backoff(&self, attempt: u32) -> Duration {
        let base = self.config.retry_delay.as_millis() as u64;
        let exponential = base * 2u64.pow(attempt.saturating_sub(1));
        let jitter = rand::random::<u64>() % 100;
        Duration::from_millis((exponential + jitter).min(30_000))
    }

    async fn do_send_request(&self, request: &RawRequest) -> Result<reqwest::Response, LlmError> {
        tracing::debug!(
            url = %request.url,
            method = ?request.method,
            "sending HTTP request"
        );

        let mut builder = match request.method {
            llm_trait::HttpMethod::Post => self.client.post(&request.url),
            llm_trait::HttpMethod::Get => self.client.get(&request.url),
            llm_trait::HttpMethod::Put => self.client.put(&request.url),
            llm_trait::HttpMethod::Delete => self.client.delete(&request.url),
        };

        for (key, value) in &request.headers {
            builder = builder.header(key.as_str(), value.as_str());
        }

        builder = builder
            .header("Content-Type", "application/json")
            .json(&request.body);

        let response = builder
            .send()
            .await
            .map_err(|e| {
                tracing::error!(error = %e, url = %request.url, "HTTP request failed");
                LlmError::llm(format!("HTTP request failed: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                status = status.as_u16(),
                url = %request.url,
                error_body = %body,
                request_body = %serde_json::to_string(&request.body).unwrap_or_default(),
                "HTTP error with full request context"
            );
            return Err(LlmError::api(status.as_u16(), body));
        }

        Ok(response)
    }
}

#[async_trait]
impl<A: RawAdapter + Send + Sync> LlmProvider for GenericProvider<A> {
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        let modes = self.adapter.supported_modes();
        if !modes.contains(&CallMode::Stream) {
            return Err(LlmError::llm("Streaming not supported by this adapter"));
        }

        let raw_request = self.adapter.build_request(&request, CallMode::Stream)?;
        self.execute_stream(raw_request).await
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let modes = self.adapter.supported_modes();

        // Prefer non-streaming mode
        if modes.contains(&CallMode::Once) {
            let raw_request = self.adapter.build_request(&request, CallMode::Once)?;
            return self.execute_once(raw_request).await;
        }

        // Fallback: stream and collect full response
        let stream = self.stream(request).await?;
        stream.collect_response().await
    }

    fn capabilities(&self) -> Capabilities {
        self.adapter.capabilities()
    }

    fn info(&self) -> ProviderInfo {
        self.adapter.info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_trait::{
        ChatMessage, FinishReason, HttpMethod, LlmBackend, StreamChunk, UsageInfo,
    };

    /// Mock adapter for testing GenericProvider
    struct MockAdapter {
        stream_supported: bool,
        once_supported: bool,
    }

    impl MockAdapter {
        fn new() -> Self {
            Self {
                stream_supported: true,
                once_supported: true,
            }
        }
    }

    #[async_trait]
    impl RawAdapter for MockAdapter {
        fn build_request(
            &self,
            _request: &ChatRequest,
            mode: CallMode,
        ) -> Result<RawRequest, LlmError> {
            Ok(RawRequest {
                url: "https://api.example.com/v1/messages".to_string(),
                method: HttpMethod::Post,
                headers: Default::default(),
                body: serde_json::json!({"model": "test"}),
                stream: mode == CallMode::Stream,
            })
        }

        async fn execute_stream(
            &self,
            _client: &reqwest::Client,
            _request: RawRequest,
        ) -> Result<ChatStream, LlmError> {
            let chunks = vec![
                Ok(StreamChunk::Text("hello".into())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".into()),
                }),
            ];
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(chunks))))
        }

        fn parse_response(&self, _body: &[u8]) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: "mock response".to_string(),
                tool_calls: vec![],
                usage: UsageInfo::default(),
                finish_reason: FinishReason::Stop,
                raw: None,
            })
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                supports_streaming: true,
                supports_tools: true,
                ..Default::default()
            }
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "mock".to_string(),
                model: "mock-model".to_string(),
                backend: LlmBackend::Custom("mock".to_string()),
                version: None,
            }
        }

        fn supported_modes(&self) -> &[CallMode] {
            &[CallMode::Stream, CallMode::Once]
        }
    }

    #[test]
    fn generic_provider_info() {
        let adapter = MockAdapter::new();
        let provider = GenericProvider::new(adapter);
        let info = provider.info();
        assert_eq!(info.name, "mock");
        assert_eq!(info.model, "mock-model");
    }

    #[test]
    fn generic_provider_capabilities() {
        let adapter = MockAdapter::new();
        let provider = GenericProvider::new(adapter);
        let caps = provider.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
    }

    #[tokio::test]
    async fn generic_provider_stream() {
        let adapter = MockAdapter::new();
        let provider = GenericProvider::new(adapter);
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let stream = provider.stream(request).await.unwrap();
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn provider_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(15));
        assert_eq!(config.request_timeout, Duration::from_secs(120));
        assert_eq!(config.max_retries, 3);
    }
}
