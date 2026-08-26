//! Qwen (Tongyi Qianwen) provider implementation.
//!
//! Qwen uses the standard OpenAI Chat Completions protocol via
//! DashScope's compatible mode. This provider wraps
//! `GenericProvider<OpenAiProtocol>` with Qwen-specific defaults.

use llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmBackend, LlmConfig, LlmError,
    LlmProvider, ProviderInfo,
};

use crate::generic::GenericProvider;
use crate::protocol::openai::OpenAiProtocol;

/// Default Qwen base URL (DashScope compatible mode)
const QWEN_DEFAULT_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// Default Qwen model
const QWEN_DEFAULT_MODEL: &str = "qwen-plus";

/// Qwen provider.
///
/// Wraps `GenericProvider<OpenAiProtocol>` with Qwen-specific defaults.
/// Implements `LlmProvider` by delegation, overriding `info()`.
pub struct QwenProvider {
    inner: GenericProvider<OpenAiProtocol>,
}

impl QwenProvider {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        let base_url = base_url.unwrap_or(QWEN_DEFAULT_BASE_URL);
        let protocol = OpenAiProtocol::new(api_key, model, Some(base_url));
        Self {
            inner: GenericProvider::new(protocol),
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        let model = if config.model.is_empty() {
            QWEN_DEFAULT_MODEL
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
impl LlmProvider for QwenProvider {
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.inner.stream(request).await
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner.chat(request).await
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn info(&self) -> ProviderInfo {
        let mut info = self.inner.info();
        info.name = "qwen".to_string();
        info.backend = LlmBackend::Custom("qwen".to_string());
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> QwenProvider {
        QwenProvider::new("sk-test", "qwen-plus", None)
    }

    #[test]
    fn qwen_provider_info() {
        let provider = make_provider();
        let info = provider.info();
        assert_eq!(info.name, "qwen");
        assert_eq!(info.model, "qwen-plus");
        assert_eq!(info.backend, LlmBackend::Custom("qwen".to_string()));
    }

    #[test]
    fn qwen_provider_capabilities() {
        let provider = make_provider();
        let caps = provider.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
    }

    #[test]
    fn qwen_provider_from_config() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "qwen-max".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            options: Default::default(),
        };
        let provider = QwenProvider::from_config(&config);
        let info = provider.info();
        assert_eq!(info.name, "qwen");
        assert_eq!(info.model, "qwen-max");
    }

    #[test]
    fn qwen_provider_custom_base_url() {
        let provider = QwenProvider::new("sk-test", "qwen-plus", Some("https://custom.api.com/v1"));
        let info = provider.info();
        assert_eq!(info.name, "qwen");
    }
}
