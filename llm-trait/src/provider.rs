//! The unified LLM provider trait.

use async_trait::async_trait;

use super::capabilities::{Capabilities, ProviderInfo};
use super::error::LlmError;
use super::request::ChatRequest;
use super::response::{ChatResponse, ChatStream};

/// LLM Provider unified interface.
///
/// The core trait exposed by the framework. All providers implement this.
/// Supports both streaming and non-streaming call modes.
///
/// # Object Safety
///
/// This trait is object-safe, so `Arc<dyn LlmProvider>` works.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Streaming call (returns chunks in real-time).
    ///
    /// Use for: chat interfaces, real-time output, long text generation.
    async fn stream(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError>;

    /// Non-streaming call (returns complete result at once).
    ///
    /// Use for: API services, batch processing, testing.
    async fn chat(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, LlmError>;

    /// Get provider capabilities.
    fn capabilities(&self) -> Capabilities;

    /// Get provider info.
    fn info(&self) -> ProviderInfo;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::LlmBackend;
    use crate::message::ChatMessage;
    use crate::response::{FinishReason, StreamChunk};
    use crate::types::UsageInfo;
    use std::sync::Arc;

    /// Mock provider for testing object safety
    struct MockProvider;

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn stream(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatStream, LlmError> {
            let chunks = vec![Ok(StreamChunk::Text("mock response".into()))];
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(chunks))))
        }

        async fn chat(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatResponse, LlmError> {
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
    }

    #[test]
    fn trait_is_object_safe() {
        let _provider: Arc<dyn LlmProvider> = Arc::new(MockProvider);
    }

    #[tokio::test]
    async fn mock_provider_chat() {
        let provider = MockProvider;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let response = provider.chat(request).await.unwrap();
        assert_eq!(response.content, "mock response");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn mock_provider_stream() {
        let provider = MockProvider;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let stream = provider.stream(request).await.unwrap();
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "mock response");
    }

    #[test]
    fn mock_provider_capabilities() {
        let provider = MockProvider;
        let caps = provider.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
    }

    #[test]
    fn mock_provider_info() {
        let provider = MockProvider;
        let info = provider.info();
        assert_eq!(info.name, "mock");
        assert_eq!(info.model, "mock-model");
    }
}
