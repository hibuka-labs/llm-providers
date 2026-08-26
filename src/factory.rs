//! Provider factory.
//!
//! Creates `LlmProvider` instances based on `LlmConfig`.

use std::sync::Arc;

use llm_trait::{LlmConfig, LlmError, LlmProvider, Protocol};

use crate::generic::GenericProvider;
use crate::protocol::anthropic::AnthropicProtocol;
use crate::protocol::openai::OpenAiProtocol;

/// Create a provider from configuration.
///
/// Routing based on protocol (explicit or inferred):
/// - protocol=OpenAi → `GenericProvider<OpenAiProtocol>`
/// - protocol=Anthropic → `GenericProvider<AnthropicProtocol>`
/// - default → `GenericProvider<OpenAiProtocol>` (fallback)
pub fn create_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    // Validate required fields
    if config.api_key.is_empty() {
        return Err(LlmError::config("LLM_API_KEY is required"));
    }
    if config.model.is_empty() {
        return Err(LlmError::config("LLM_MODEL is required"));
    }
    if config.base_url.is_empty() {
        return Err(LlmError::config("LLM_BASE_URL is required"));
    }

    let protocol = config.resolve_protocol();

    match protocol {
        Protocol::OpenAi | Protocol::OpenAiResponses => {
            let protocol_impl = OpenAiProtocol::from_config(config);
            let provider = GenericProvider::new(protocol_impl);
            Ok(Arc::new(provider))
        }
        Protocol::Anthropic => {
            let protocol_impl = AnthropicProtocol::from_config(config);
            let provider = GenericProvider::new(protocol_impl);
            Ok(Arc::new(provider))
        }
    }
}

/// Convenience: create a provider with minimal parameters.
pub fn create(
    api_key: &str,
    model: &str,
    base_url: &str,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let config = LlmConfig {
        protocol: None,
        api_key: api_key.to_string(),
        model: model.to_string(),
        base_url: base_url.to_string(),
        options: std::collections::HashMap::new(),
    };
    create_provider(&config)
}

/// Create a provider from environment variables.
pub fn from_env() -> Result<Arc<dyn LlmProvider>, LlmError> {
    let config = LlmConfig::from_env()?;
    create_provider(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_provider_openai() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "openai");
        assert_eq!(info.model, "gpt-4o");
    }

    #[test]
    fn create_provider_anthropic() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "claude-sonnet".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "anthropic");
        assert_eq!(info.model, "claude-sonnet");
    }

    #[test]
    fn create_provider_explicit_protocol() {
        let config = LlmConfig {
            protocol: Some("anthropic".to_string()),
            api_key: "sk-test".to_string(),
            model: "test-model".to_string(),
            base_url: "https://custom.api.com".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "anthropic");
    }

    #[test]
    fn create_convenience() {
        let provider = create("sk-test", "gpt-4o", "https://api.openai.com/v1").unwrap();
        let info = provider.info();
        assert_eq!(info.name, "openai");
    }
}
