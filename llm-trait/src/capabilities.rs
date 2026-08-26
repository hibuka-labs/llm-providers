//! Provider capabilities and info types.

use super::backend::LlmBackend;

/// Provider capability description.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    /// Supports streaming responses
    pub supports_streaming: bool,
    /// Supports tool calling
    pub supports_tools: bool,
    /// Supports vision (images)
    pub supports_vision: bool,
    /// Supports thinking/reasoning
    pub supports_thinking: bool,
    /// Maximum context token count
    pub max_context_tokens: Option<u32>,
    /// Maximum output token count
    pub max_output_tokens: Option<u32>,
}

/// Provider information.
#[derive(Clone, Debug)]
pub struct ProviderInfo {
    /// Provider name (e.g. "openai", "anthropic")
    pub name: String,
    /// Model name (e.g. "gpt-4o", "claude-sonnet")
    pub model: String,
    /// Backend type
    pub backend: LlmBackend,
    /// Version info (optional)
    pub version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_default() {
        let caps = Capabilities::default();
        assert!(!caps.supports_streaming);
        assert!(!caps.supports_tools);
        assert!(!caps.supports_vision);
        assert!(!caps.supports_thinking);
        assert!(caps.max_context_tokens.is_none());
        assert!(caps.max_output_tokens.is_none());
    }

    #[test]
    fn capabilities_clone() {
        let caps = Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: true,
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
        };
        let cloned = caps.clone();
        assert!(cloned.supports_streaming);
        assert!(cloned.supports_tools);
        assert!(!cloned.supports_vision);
        assert!(cloned.supports_thinking);
        assert_eq!(cloned.max_context_tokens, Some(128_000));
        assert_eq!(cloned.max_output_tokens, Some(16_384));
    }

    #[test]
    fn provider_info_debug() {
        let info = ProviderInfo {
            name: "openai".to_string(),
            model: "gpt-4o".to_string(),
            backend: LlmBackend::OpenAi,
            version: None,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("openai"));
        assert!(debug.contains("gpt-4o"));
    }
}
