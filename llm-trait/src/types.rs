//! Usage info types.

/// Token usage information for an LLM call.
#[derive(Clone, Debug, Default)]
pub struct UsageInfo {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    /// Tokens used for reasoning / thinking (DeepSeek, OpenAI o-series, etc.).
    pub reasoning_tokens: Option<u32>,
}
