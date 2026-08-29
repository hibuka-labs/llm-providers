//! LLM layer unified error type.
//!
//! Defined in `llm-trait` so downstream runtimes can convert it into their own
//! error type via `From<LlmError>`. This keeps `llm-trait` lightweight — it does
//! not depend on any consumer crate.

/// LLM layer unified error type.
///
/// Covers configuration errors, API errors, general LLM errors,
/// and stream parsing errors. Designed to be converted into a
/// consumer's own error type by the runtime layer.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    /// Configuration error (missing API key, invalid model name, etc.)
    #[error("Config error: {0}")]
    Config(String),

    /// API returned an error (with HTTP status code).
    #[error("API error {status}: {message}")]
    LlmApi { status: u16, message: String },

    /// General LLM error (network timeout, connection failure, etc.)
    #[error("LLM error: {0}")]
    Llm(String),

    /// Stream parsing error (SSE format issues, etc.)
    #[error("Stream error: {0}")]
    Stream(String),
}

impl LlmError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn llm(msg: impl Into<String>) -> Self {
        Self::Llm(msg.into())
    }

    pub fn stream(msg: impl Into<String>) -> Self {
        Self::Stream(msg.into())
    }

    pub fn api(status: u16, message: impl Into<String>) -> Self {
        Self::LlmApi {
            status,
            message: message.into(),
        }
    }

    /// HTTP status code, if this error came from an API response.
    ///
    /// Useful for deciding whether to retry or how to surface the failure.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::LlmApi { status, .. } => Some(*status),
            _ => None,
        }
    }
}

// Convenience From impls for common error types.

impl From<serde_json::Error> for LlmError {
    fn from(e: serde_json::Error) -> Self {
        LlmError::Llm(format!("JSON error: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_config() {
        let err = LlmError::config("missing key");
        assert_eq!(err.to_string(), "Config error: missing key");
    }

    #[test]
    fn display_api() {
        let err = LlmError::api(401, "unauthorized");
        assert_eq!(err.to_string(), "API error 401: unauthorized");
        assert_eq!(err.status(), Some(401));
    }

    #[test]
    fn status_is_none_for_non_api_errors() {
        assert_eq!(LlmError::config("missing key").status(), None);
        assert_eq!(LlmError::llm("timeout").status(), None);
        assert_eq!(LlmError::stream("bad SSE").status(), None);
    }

    #[test]
    fn display_llm() {
        let err = LlmError::llm("timeout");
        assert_eq!(err.to_string(), "LLM error: timeout");
    }

    #[test]
    fn display_stream() {
        let err = LlmError::stream("bad SSE");
        assert_eq!(err.to_string(), "Stream error: bad SSE");
    }

    #[test]
    fn from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let llm_err: LlmError = json_err.into();
        assert!(llm_err.to_string().contains("JSON error"));
    }

    #[test]
    fn is_clone() {
        let err = LlmError::llm("test");
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }
}
