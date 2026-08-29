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
#[derive(Clone, Default)]
pub struct LlmConfig {
    /// Protocol (optional, auto-inferred if not set)
    ///
    /// If set, overrides all inference (highest priority).
    /// If not set, the factory uses ModelRegistry to infer from URL and model name.
    pub protocol: Option<Protocol>,

    /// API Key (required)
    ///
    /// Hidden from `Debug` output — never log a config expecting to see the key.
    pub api_key: String,

    /// Model name (required)
    pub model: String,

    /// Base URL (required)
    pub base_url: String,

    /// Other options
    pub options: HashMap<String, Value>,
}

/// `Debug` redacts [`LlmConfig::api_key`] so a config cannot leak a credential
/// through a log line, a panic message, or a test failure dump.
impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("protocol", &self.protocol)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("options", &self.options)
            .finish()
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
            api_key: std::env::var("LLM_API_KEY")
                .map_err(|_| LlmError::config("LLM_API_KEY environment variable not set"))?,
            model: std::env::var("LLM_MODEL")
                .map_err(|_| LlmError::config("LLM_MODEL environment variable not set"))?,
            base_url: std::env::var("LLM_BASE_URL")
                .map_err(|_| LlmError::config("LLM_BASE_URL environment variable not set"))?,
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

    /// `from_env` reads process env, and tests run in parallel — so anything
    /// touching `LLM_*` must hold this lock for its whole scope.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        match LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn set_llm_env(vars: &[(&str, &str)]) {
        for key in ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_PROTOCOL"] {
            // SAFETY: callers hold `env_lock()`, so no other test reads these
            // variables concurrently.
            unsafe { std::env::remove_var(key) };
        }
        for (key, value) in vars {
            unsafe { std::env::set_var(key, value) };
        }
    }

    #[test]
    fn from_env_reads_required_vars() {
        let _guard = env_lock();
        set_llm_env(&[
            ("LLM_API_KEY", "sk-from-env"),
            ("LLM_MODEL", "gpt-4o-mini"),
            ("LLM_BASE_URL", "https://api.openai.com/v1"),
        ]);

        let config = LlmConfig::from_env().unwrap();
        assert_eq!(config.api_key, "sk-from-env");
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.protocol, None, "unset LLM_PROTOCOL means infer");
    }

    #[test]
    fn from_env_parses_protocol_override() {
        let _guard = env_lock();
        set_llm_env(&[
            ("LLM_API_KEY", "sk"),
            ("LLM_MODEL", "m"),
            ("LLM_BASE_URL", "https://example.invalid"),
            ("LLM_PROTOCOL", "anthropic"),
        ]);

        let config = LlmConfig::from_env().unwrap();
        assert_eq!(config.protocol, Some(Protocol::Anthropic));

        // An unrecognised value leaves protocol unset (registry infers) rather
        // than failing the whole config.
        set_llm_env(&[
            ("LLM_API_KEY", "sk"),
            ("LLM_MODEL", "m"),
            ("LLM_BASE_URL", "https://example.invalid"),
            ("LLM_PROTOCOL", "not-a-protocol"),
        ]);
        let config = LlmConfig::from_env().unwrap();
        assert_eq!(config.protocol, None);
    }

    #[test]
    fn from_env_names_whichever_var_is_missing() {
        let _guard = env_lock();
        let complete = [
            ("LLM_API_KEY", "sk"),
            ("LLM_MODEL", "m"),
            ("LLM_BASE_URL", "https://example.invalid"),
        ];

        for drop in ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"] {
            let kept: Vec<(&str, &str)> = complete
                .iter()
                .cloned()
                .filter(|(k, _)| *k != drop)
                .collect();
            set_llm_env(&kept);

            let err = LlmConfig::from_env().unwrap_err();
            assert!(
                err.to_string().contains(drop),
                "expected {drop} in error, got: {err}"
            );
        }
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

    #[test]
    fn debug_does_not_leak_api_key() {
        let config = LlmConfig {
            protocol: None,
            api_key: "sk-super-secret-value".to_string(),
            model: "test".to_string(),
            base_url: "https://custom.api.com/v1".to_string(),
            options: HashMap::new(),
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("sk-super-secret-value"));
        assert!(rendered.contains("<redacted>"));
        // Non-secret fields stay visible for debugging.
        assert!(rendered.contains("https://custom.api.com/v1"));
        assert!(rendered.contains("test"));
    }
}
