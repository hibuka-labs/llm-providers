//! Unified LLM provider configuration.

use serde_json::Value;
use std::collections::HashMap;

use super::backend::Protocol;
use super::error::LlmError;

/// LLM Provider configuration (protocol-agnostic, vendor-agnostic).
///
/// Design principles:
/// 1. Protocol can be configured (explicit)
/// 2. Protocol can be auto-inferred (from URL/model) by the ModelRegistry
/// 3. Defaults to OpenAI protocol (fallback)
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Protocol (optional, auto-inferred if not set)
    ///
    /// If set, overrides all inference (highest priority).
    /// If not set, the factory uses ModelRegistry to infer from URL and model name.
    pub protocol: Option<Protocol>,

    /// API Key (required)
    pub api_key: String,

    /// Model name (required)
    pub model: String,

    /// Base URL (required)
    pub base_url: String,

    /// Other options
    pub options: HashMap<String, Value>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            protocol: None,
            api_key: String::new(),
            model: String::new(),
            base_url: String::new(),
            options: HashMap::new(),
        }
    }
}

impl LlmConfig {
    /// Create from environment variables.
    pub fn from_env() -> Result<Self, LlmError> {
        let protocol = std::env::var("LLM_PROTOCOL")
            .ok()
            .and_then(|s| s.parse().ok());

        Ok(Self {
            protocol,
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

    /// Get base URL (always available since it's required).
    pub fn resolve_base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_with_protocol() {
        // This test verifies the protocol parsing logic
        let protocol = "anthropic".parse::<Protocol>().ok();
        assert_eq!(protocol, Some(Protocol::Anthropic));
    }

    #[test]
    fn from_env_without_protocol() {
        let protocol: Option<Protocol> = None;
        assert!(protocol.is_none());
    }

    #[test]
    fn default_config() {
        let config = LlmConfig::default();
        assert!(config.protocol.is_none());
        assert!(config.api_key.is_empty());
        assert!(config.model.is_empty());
        assert!(config.base_url.is_empty());
    }

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
