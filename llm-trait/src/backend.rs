//! LLM provider backend and protocol type definitions.

/// LLM Provider backend type (internal use).
///
/// Used to identify different provider implementations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmBackend {
    /// OpenAI Chat Completions API
    OpenAi,
    /// OpenAI Responses API
    OpenAiResponses,
    /// Anthropic Claude API
    Anthropic,
    /// Custom backend
    Custom(String),
}

impl LlmBackend {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "openai" => Self::OpenAi,
            "openai-responses" | "responses" => Self::OpenAiResponses,
            "anthropic" => Self::Anthropic,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenAiResponses => "openai-responses",
            Self::Anthropic => "anthropic",
            Self::Custom(s) => s,
        }
    }
}

/// Protocol type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// OpenAI compatible protocol (Chat Completions)
    OpenAi,
    /// OpenAI Responses API
    OpenAiResponses,
    /// Anthropic protocol
    Anthropic,
}

impl Protocol {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" | "openai-chat" => Some(Self::OpenAi),
            "openai-responses" | "responses" => Some(Self::OpenAiResponses),
            "anthropic" | "claude" => Some(Self::Anthropic),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenAiResponses => "openai-responses",
            Self::Anthropic => "anthropic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_from_str_case_insensitive() {
        assert_eq!(LlmBackend::from_str("openai"), LlmBackend::OpenAi);
        assert_eq!(LlmBackend::from_str("OpenAI"), LlmBackend::OpenAi);
        assert_eq!(LlmBackend::from_str("OPENAI"), LlmBackend::OpenAi);
    }

    #[test]
    fn backend_from_str_anthropic() {
        assert_eq!(LlmBackend::from_str("anthropic"), LlmBackend::Anthropic);
        assert_eq!(LlmBackend::from_str("Anthropic"), LlmBackend::Anthropic);
    }

    #[test]
    fn backend_from_str_responses() {
        assert_eq!(
            LlmBackend::from_str("openai-responses"),
            LlmBackend::OpenAiResponses
        );
        assert_eq!(
            LlmBackend::from_str("responses"),
            LlmBackend::OpenAiResponses
        );
    }

    #[test]
    fn backend_from_str_custom() {
        assert_eq!(
            LlmBackend::from_str("ollama"),
            LlmBackend::Custom("ollama".to_string())
        );
        assert_eq!(
            LlmBackend::from_str("deepseek"),
            LlmBackend::Custom("deepseek".to_string())
        );
    }

    #[test]
    fn backend_as_str_roundtrip() {
        assert_eq!(LlmBackend::OpenAi.as_str(), "openai");
        assert_eq!(LlmBackend::Anthropic.as_str(), "anthropic");
        assert_eq!(LlmBackend::OpenAiResponses.as_str(), "openai-responses");
        assert_eq!(LlmBackend::Custom("ollama".into()).as_str(), "ollama");
    }

    #[test]
    fn protocol_from_str() {
        assert_eq!(Protocol::from_str("openai"), Some(Protocol::OpenAi));
        assert_eq!(Protocol::from_str("openai-chat"), Some(Protocol::OpenAi));
        assert_eq!(Protocol::from_str("anthropic"), Some(Protocol::Anthropic));
        assert_eq!(Protocol::from_str("claude"), Some(Protocol::Anthropic));
        assert_eq!(
            Protocol::from_str("openai-responses"),
            Some(Protocol::OpenAiResponses)
        );
        assert_eq!(
            Protocol::from_str("responses"),
            Some(Protocol::OpenAiResponses)
        );
        assert_eq!(Protocol::from_str("unknown"), None);
    }

    #[test]
    fn protocol_as_str() {
        assert_eq!(Protocol::OpenAi.as_str(), "openai");
        assert_eq!(Protocol::Anthropic.as_str(), "anthropic");
        assert_eq!(Protocol::OpenAiResponses.as_str(), "openai-responses");
    }
}
