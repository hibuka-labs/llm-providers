//! Protocol type definitions.

use std::fmt;
use std::str::FromStr;

/// Wire protocol type.
///
/// Describes the API format used to communicate with the LLM provider.
/// This is distinct from the provider name (e.g., "deepseek" uses OpenAi protocol).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// OpenAI Chat Completions API
    OpenAi,
    /// OpenAI Responses API (future)
    OpenAiResponses,
    /// Anthropic Messages API
    Anthropic,
}

/// Error returned when parsing an invalid protocol string.
#[derive(Debug, Clone)]
pub struct ProtocolParseError(String);

impl fmt::Display for ProtocolParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown protocol: '{}'", self.0)
    }
}

impl std::error::Error for ProtocolParseError {}

impl FromStr for Protocol {
    type Err = ProtocolParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" | "openai-chat" => Ok(Self::OpenAi),
            "openai-responses" | "responses" => Ok(Self::OpenAiResponses),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            _ => Err(ProtocolParseError(s.to_string())),
        }
    }
}

impl Protocol {
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
    fn protocol_from_str() {
        assert_eq!("openai".parse::<Protocol>().unwrap(), Protocol::OpenAi);
        assert_eq!("openai-chat".parse::<Protocol>().unwrap(), Protocol::OpenAi);
        assert_eq!("anthropic".parse::<Protocol>().unwrap(), Protocol::Anthropic);
        assert_eq!("claude".parse::<Protocol>().unwrap(), Protocol::Anthropic);
        assert_eq!(
            "openai-responses".parse::<Protocol>().unwrap(),
            Protocol::OpenAiResponses
        );
        assert_eq!(
            "responses".parse::<Protocol>().unwrap(),
            Protocol::OpenAiResponses
        );
        assert!("unknown".parse::<Protocol>().is_err());
    }

    #[test]
    fn protocol_as_str() {
        assert_eq!(Protocol::OpenAi.as_str(), "openai");
        assert_eq!(Protocol::Anthropic.as_str(), "anthropic");
        assert_eq!(Protocol::OpenAiResponses.as_str(), "openai-responses");
    }

    #[test]
    fn protocol_from_str_case_insensitive() {
        assert_eq!("OpenAI".parse::<Protocol>().unwrap(), Protocol::OpenAi);
        assert_eq!(
            "ANTHROPIC".parse::<Protocol>().unwrap(),
            Protocol::Anthropic
        );
    }
}
