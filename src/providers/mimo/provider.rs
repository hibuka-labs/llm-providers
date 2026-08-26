//! Mimo provider implementation.
//!
//! Mimo is an Anthropic-compatible API provider. This provider wraps
//! `GenericProvider<AnthropicProtocol>` with Mimo-specific configuration.

use llm_trait::{ChatRequest, ChatResponse, ChatStream, LlmBackend, LlmConfig, LlmError, LlmProvider};

use crate::generic::GenericProvider;
use crate::protocol::anthropic::AnthropicProtocol;

/// Default Mimo base URL for Anthropic protocol
const MIMO_DEFAULT_BASE_URL: &str = "https://api.xiaomimimo.com/anthropic";

/// Mimo provider.
///
/// Wraps `GenericProvider<AnthropicProtocol>` with Mimo-specific defaults.
/// Implements `LlmProvider` by delegation.
pub struct MimoProvider {
    inner: GenericProvider<AnthropicProtocol>,
}

impl MimoProvider {
    pub fn new(api_key: &str, model: &str, base_url: Option<&str>) -> Self {
        let base_url = base_url.unwrap_or(MIMO_DEFAULT_BASE_URL);
        let protocol = AnthropicProtocol::new(api_key, model, Some(base_url));
        Self {
            inner: GenericProvider::new(protocol),
        }
    }

    pub fn from_config(config: &LlmConfig) -> Self {
        let protocol = AnthropicProtocol::new(&config.api_key, &config.model, Some(&config.base_url));
        Self {
            inner: GenericProvider::new(protocol),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for MimoProvider {
    async fn stream(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        self.inner.stream(request).await
    }

    async fn chat(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        self.inner.chat(request).await
    }

    fn capabilities(&self) -> llm_trait::Capabilities {
        self.inner.capabilities()
    }

    fn info(&self) -> llm_trait::ProviderInfo {
        let mut info = self.inner.info();
        info.name = "mimo".to_string();
        info.backend = LlmBackend::Custom("mimo".to_string());
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> MimoProvider {
        MimoProvider::new("tp-test", "mimo-v2.5-pro", None)
    }

    #[test]
    fn mimo_provider_info() {
        let provider = make_provider();
        let info = provider.info();
        assert_eq!(info.name, "mimo");
        assert_eq!(info.model, "mimo-v2.5-pro");
        assert_eq!(info.backend, LlmBackend::Custom("mimo".to_string()));
    }

    #[test]
    fn mimo_provider_capabilities() {
        let provider = make_provider();
        let caps = provider.capabilities();
        assert!(caps.supports_streaming);
        assert!(caps.supports_tools);
        assert!(caps.supports_vision);
        assert!(caps.supports_thinking);
    }

    #[test]
    fn mimo_provider_from_config() {
        let config = LlmConfig {
            protocol: None,
            api_key: "tp-test".to_string(),
            model: "mimo-v2.5-pro".to_string(),
            base_url: "https://api.xiaomimimo.com/anthropic".to_string(),
            options: Default::default(),
        };
        let provider = MimoProvider::from_config(&config);
        let info = provider.info();
        assert_eq!(info.name, "mimo");
        assert_eq!(info.model, "mimo-v2.5-pro");
    }

    #[test]
    fn mimo_provider_custom_base_url() {
        let provider = MimoProvider::new("tp-test", "mimo-v2.5-pro", Some("https://custom.api.com"));
        let info = provider.info();
        assert_eq!(info.name, "mimo");
    }
}
