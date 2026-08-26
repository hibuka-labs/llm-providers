//! Unified LLM provider configuration.

use serde_json::Value;
use std::collections::HashMap;

use super::backend::Protocol;
use super::error::LlmError;

/// LLM Provider configuration (protocol-agnostic, vendor-agnostic).
///
/// Design principles:
/// 1. Protocol can be configured (explicit)
/// 2. Protocol can be auto-inferred (from URL/model)
/// 3. Defaults to OpenAI protocol (fallback)
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Protocol (optional, auto-inferred)
    ///
    /// Examples: "openai", "anthropic"
    /// If not set, inferred automatically from URL or model name
    pub protocol: Option<String>,

    /// API Key (required)
    pub api_key: String,

    /// Model name (required)
    pub model: String,

    /// Base URL (required)
    pub base_url: String,

    /// Other options
    pub options: HashMap<String, Value>,
}

impl LlmConfig {
    /// Create from environment variables.
    pub fn from_env() -> Result<Self, LlmError> {
        Ok(Self {
            protocol: std::env::var("LLM_PROTOCOL").ok(),
            api_key: std::env::var("LLM_API_KEY").map_err(|_| {
                LlmError::config("LLM_API_KEY environment variable not set")
            })?,
            model: std::env::var("LLM_MODEL").map_err(|_| {
                LlmError::config("LLM_MODEL environment variable not set")
            })?,
            base_url: std::env::var("LLM_BASE_URL").map_err(|_| {
                LlmError::config("LLM_BASE_URL environment variable not set")
            })?,
            options: HashMap::new(),
        })
    }

    /// Resolve protocol (explicit config > inference > default).
    ///
    /// Inference priority (highest to lowest):
    /// 1. Explicit config - `protocol` field present
    /// 2. URL signature - `base_url` contains specific keywords
    /// 3. Model name - model starts with specific prefix
    /// 4. Default - OpenAI protocol
    pub fn resolve_protocol(&self) -> Protocol {
        // 1. Explicit config (highest priority)
        if let Some(ref protocol_str) = self.protocol {
            return Protocol::from_str(protocol_str).unwrap_or_else(|| {
                tracing::warn!(
                    "Unknown protocol '{}', falling back to OpenAI",
                    protocol_str
                );
                Protocol::OpenAi
            });
        }

        // 2. URL signature matching
        let url_lower = self.base_url.to_lowercase();
        if url_lower.contains("anthropic") || url_lower.contains("claude") {
            return Protocol::Anthropic;
        }

        // 3. Model name matching
        let model_lower = self.model.to_lowercase();
        if model_lower.starts_with("claude") || model_lower.starts_with("anthropic") {
            return Protocol::Anthropic;
        }

        // 4. Default OpenAI (fallback)
        Protocol::OpenAi
    }

    /// Get base URL (always available since it's required).
    pub fn resolve_base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_protocol ──

    #[test]
    fn resolve_protocol_explicit_overrides_all() {
        let config = LlmConfig {
            protocol: Some("anthropic".to_string()),
            api_key: "sk-test".to_string(),
            model: "test".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::Anthropic);
    }

    #[test]
    fn resolve_protocol_url_anthropic() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "test".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::Anthropic);
    }

    #[test]
    fn resolve_protocol_url_claude() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "test".to_string(),
            base_url: "https://claude.example.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::Anthropic);
    }

    #[test]
    fn resolve_protocol_model_claude() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::Anthropic);
    }

    #[test]
    fn resolve_protocol_default_openai() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "deepseek-chat".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::OpenAi);
    }

    #[test]
    fn resolve_protocol_unknown_falls_back_to_openai() {
        let config = LlmConfig {
            protocol: Some("unknown-protocol".to_string()),
            api_key: "sk-test".to_string(),
            model: "test".to_string(),
            base_url: "https://api.example.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_protocol(), Protocol::OpenAi);
    }

    // ── resolve_base_url ──

    #[test]
    fn resolve_base_url_returns_configured_url() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-test".to_string(),
            model: "test".to_string(),
            base_url: "https://custom.api.com/v1".to_string(),
            options: HashMap::new(),
        };
        assert_eq!(config.resolve_base_url(), "https://custom.api.com/v1");
    }
}
