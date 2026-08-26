//! Reasoning/thinking configuration types.

/// Reasoning/thinking configuration, unifying reasoning/thinking parameters across vendors.
#[derive(Debug, Clone, Default)]
pub struct ReasoningConfig {
    /// Whether to enable reasoning/thinking process
    pub enabled: Option<bool>,
    /// Thinking budget (token count limit)
    pub budget_tokens: Option<u64>,
    /// Reasoning intensity/depth (semantics vary by vendor)
    pub effort: Option<ReasoningEffort>,
}

/// Reasoning intensity/depth enumeration.
#[derive(Debug, Clone, Default)]
pub enum ReasoningEffort {
    #[default]
    None,
    Low,
    Medium,
    High,
    XHigh,
}
