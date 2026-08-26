//! DeepSeek provider implementation.
//!
//! DeepSeek uses the standard OpenAI Chat Completions protocol.
//! This provider wraps `GenericProvider<OpenAiProtocol>` with
//! DeepSeek-specific defaults.

use llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmBackend, LlmConfig, LlmError,
    LlmProvider, ProviderInfo,
};

use crate::generic::GenericProvider;
use crate::protocol::openai::OpenAiProtocol;

/// Default DeepSeek base URL
const DEEPSEEK_DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";

/// Default DeepSeek model
const DEEPSEEK_DEFAULT_MODEL: &str = "deepseek-chat";

/// DeepSeek provider.
///
/// Wraps `GenericProvider<OpenAiProtocol>` with DeepSeek-specific defaults.
/// Implements `LlmProvider` by delegation, overriding `info()` and `capabilities()`.
pub struct DeepSeekProvider {
    inner: GenericProvider<OpenAiProtocol>,
}

impl DeepSeekProvider {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        let base_url = base_url.unwrap_or(DEEPSEEK_DEFAULT_BASE_URL);
        let protocol = OpenAiProtocol::new(api_key, model, Some(base_url));
        Self {
            inner: GenericProvider::new(protocol),
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        let model = if config.model.is_empty() {
            DEEPSEEK_DEFAULT_MODEL
        } else {
            &config.model
        };
        let protocol = OpenAiProtocol::new(&config.api_key, model, Some(&config.base_url));
        Self {
            inner: GenericProvider::new(protocol),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for DeepSeekProvider {
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.inner.stream(request).await
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner.chat(request).await
    }

    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.capabilities();
        // DeepSeek supports extended thinking
        caps.supports_thinking = true;
        caps
    }

    fn info(&self) -> ProviderInfo {
        let mut info = self.inner.info();
        info.name = "deepseek".to_string();
        info.backend = LlmBackend::Custom("deepseek".to_string());
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> DeepSeekProvider {
        DeepSeekProvider::new("sk-test", "deepseek-chat", None)
    }

    #[test]
    fn deepseek_provider_info() {
        let provider = make_provider();
        let info = provider.info();
        assert_eq!(info.name, "deepseek");
        assert_eq!(info.model, "deepseek-chat");
        assert_eq!(info.backend, LlmBackend::Custom("deepseek".to_string()));
    }

    #[test]
    fn deepseek_provider_capabilities() {
        let provider = make_provider();
        let caps = provider.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_thinking);
    }

    #[test]
    fn deepseek_provider_from_config() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "deepseek-reasoner".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            options: Default::default(),
        };
        let provider = DeepSeekProvider::from_config(&config);
        let info = provider.info();
        assert_eq!(info.name, "deepseek");
        assert_eq!(info.model, "deepseek-reasoner");
    }

    #[test]
    fn deepseek_provider_custom_base_url() {
        let provider =
            DeepSeekProvider::new("sk-test", "deepseek-chat", Some("https://custom.api.com/v1"));
        let info = provider.info();
        assert_eq!(info.name, "deepseek");
    }
}
