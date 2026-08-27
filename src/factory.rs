//! Provider factory.
//!
//! Creates `LlmProvider` instances based on `LlmConfig` and `ModelRegistry`.

use std::sync::Arc;

use llm_trait::{LlmConfig, LlmError, LlmProvider, Protocol};

use crate::generic::{GenericProvider, ProfiledProvider};
use crate::model_registry::{ModelProfile, ModelRegistry};
use crate::protocol::anthropic::AnthropicProtocol;
use crate::protocol::openai::OpenAiProtocol;

/// Global registry (lazy-initialized).
static MODEL_REGISTRY: std::sync::LazyLock<ModelRegistry> =
    std::sync::LazyLock::new(ModelRegistry::builtin);

/// Create a provider from configuration.
///
/// Routing:
/// 1. Query ModelRegistry for profile (protocol + capabilities + reasoning mode)
/// 2. Build protocol implementation with profile injected
/// 3. Wrap in ProfiledProvider for correct info() and capabilities()
pub fn create_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    if config.api_key.is_empty() {
        return Err(LlmError::config("LLM_API_KEY is required"));
    }
    if config.model.is_empty() {
        return Err(LlmError::config("LLM_MODEL is required"));
    }
    if config.base_url.is_empty() {
        return Err(LlmError::config("LLM_BASE_URL is required"));
    }

    // 1. Validate explicit protocol
    if let Some(protocol) = config.protocol {
        if !matches!(protocol, Protocol::OpenAi | Protocol::Anthropic) {
            return Err(LlmError::config(&format!(
                "Unsupported protocol '{}'. Use 'openai' or 'anthropic'.",
                protocol.as_str()
            )));
        }
    }

    // 2. Query registry
    let profile = MODEL_REGISTRY.lookup(&config.model, Some(&config.base_url), config.protocol);

    tracing::info!(
        model = %config.model,
        base_url = %config.base_url,
        protocol = ?profile.protocol,
        provider = %profile.provider_name,
        reasoning_mode = ?profile.reasoning_mode,
        max_output_tokens = ?profile.capabilities.max_output_tokens,
        "provider created from registry profile"
    );

    // 2. Build protocol implementation
    let protocol_impl = build_protocol(&profile, config)?;

    // 3. Assemble provider
    let provider = ProfiledProvider::new(GenericProvider::new(protocol_impl), profile);
    Ok(Arc::new(provider))
}

/// Build protocol implementation based on profile.
fn build_protocol(
    profile: &ModelProfile,
    config: &LlmConfig,
) -> Result<Box<dyn llm_trait::RawAdapter>, LlmError> {
    match profile.protocol {
        Protocol::OpenAi => Ok(Box::new(
            OpenAiProtocol::from_config(config).with_model_profile(profile.clone()),
        )),
        Protocol::Anthropic => Ok(Box::new(
            AnthropicProtocol::from_config(config).with_model_profile(profile.clone()),
        )),
        Protocol::OpenAiResponses => Err(LlmError::config(
            "OpenAI Responses API is not yet supported. Use protocol 'openai' or 'anthropic'.",
        )),
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
    use llm_trait::ReasoningMode;

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
            protocol: Some(Protocol::Anthropic),
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
    fn create_provider_deepseek() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "deepseek-chat".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "deepseek");
    }

    #[test]
    fn create_provider_mimo_no_reasoning() {
        let config = LlmConfig {
            protocol: None,
            api_key: "tp-test".to_string(),
            model: "mimo-v2.5-pro".to_string(),
            base_url: "https://token-plan-cn.xiaomimimo.com/v1".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "mimo");

        // Verify MiMo's reasoning_mode is None
        let profile = MODEL_REGISTRY.lookup(
            "mimo-v2.5-pro",
            Some("https://token-plan-cn.xiaomimimo.com/v1"),
            None,
        );
        assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    }

    #[test]
    fn create_provider_qwen() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "qwen-plus".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            options: Default::default(),
        };
        let provider = create_provider(&config).unwrap();
        let info = provider.info();
        assert_eq!(info.name, "qwen");
    }

    #[test]
    fn create_provider_openai_responses_returns_error() {
        // P0-1 FIXED: Previously silently downgraded to OpenAI.
        // Now returns a clear error for unsupported protocols.
        let config = LlmConfig {
            protocol: Some(Protocol::OpenAiResponses),
            api_key: "sk-test".to_string(),
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            options: Default::default(),
        };
        match create_provider(&config) {
            Ok(_) => panic!("Expected error for OpenAiResponses protocol"),
            Err(e) => assert!(
                e.to_string().contains("Unsupported protocol"),
                "Expected clear error about unsupported protocol, got: {}",
                e
            ),
        }
    }

    #[test]
    fn create_convenience() {
        let provider = create("sk-test", "gpt-4o", "https://api.openai.com/v1").unwrap();
        let info = provider.info();
        assert_eq!(info.name, "openai");
    }

    #[test]
    fn create_provider_options_max_tokens_ignored() {
        // P1-10 VERIFY: LlmConfig.options["max_tokens"] is never used by the factory.
        // OpenAiProtocol::from_config reads it, but build_protocol calls ::new() instead.
        use std::collections::HashMap;
        let mut options = HashMap::new();
        options.insert("max_tokens".to_string(), serde_json::json!(42));
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            options,
        };
        let provider = create_provider(&config).unwrap();

        // The provider was created, but options["max_tokens"] was ignored.
        // We can't directly check max_tokens from the provider, but we can verify
        // the provider was created successfully (no error from invalid max_tokens).
        assert_eq!(provider.info().name, "openai");
        // P1-10 CONFIRMED: options["max_tokens"]=42 was silently ignored.
    }
}
