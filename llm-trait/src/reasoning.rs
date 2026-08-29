//! Reasoning/thinking configuration types.

/// How a model expresses reasoning/thinking capability.
///
/// Each model supports one or more reasoning modes.
/// The Protocol layer uses this to decide which fields to include in the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningMode {
    /// No reasoning support (don't send any reasoning fields)
    None,
    /// OpenAI-style effort parameter (`reasoning_effort: "low"|"medium"|"high"`)
    Effort,
    /// Anthropic-style thinking block (`thinking: {type: "enabled", budget_tokens: N}`)
    Thinking,
    /// Both modes supported (protocol decides which to use)
    Both,
}

/// Resolved reasoning specification — what to actually send to the API.
///
/// Derived from `ReasoningConfig` (user intent) + `ReasoningMode` (model capability).
#[derive(Debug, Clone)]
pub enum ReasoningSpec {
    /// Don't send any reasoning fields
    None,
    /// Send OpenAI-style effort parameter
    Effort(ReasoningEffort),
    /// Send Anthropic-style thinking block
    Thinking { budget_tokens: u64 },
}

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

impl ReasoningConfig {
    /// Convert user intent (ReasoningConfig) + model capability (ReasoningMode)
    /// into a concrete ReasoningSpec to send to the API.
    pub fn to_spec(&self, mode: ReasoningMode) -> ReasoningSpec {
        // If explicitly disabled, skip reasoning regardless of mode
        if self.enabled == Some(false) {
            return ReasoningSpec::None;
        }

        match mode {
            ReasoningMode::None => ReasoningSpec::None,
            ReasoningMode::Effort => {
                // Only accept Effort-type parameters
                self.effort
                    .as_ref()
                    .filter(|e| !matches!(e, ReasoningEffort::None))
                    .map(|e| ReasoningSpec::Effort(e.clone()))
                    .unwrap_or(ReasoningSpec::None)
            }
            ReasoningMode::Thinking => {
                // Only accept Thinking-type parameters
                ReasoningSpec::Thinking {
                    budget_tokens: self.budget_tokens.unwrap_or(2048),
                }
            }
            ReasoningMode::Both => {
                // Prefer Thinking over Effort.
                // Thinking is more expressive; Anthropic only handles Thinking.
                // If budget_tokens is provided, use Thinking; otherwise fall back to Effort.
                if let Some(budget_tokens) = self.budget_tokens {
                    ReasoningSpec::Thinking { budget_tokens }
                } else if let Some(effort) = &self.effort {
                    if !matches!(effort, ReasoningEffort::None) {
                        return ReasoningSpec::Effort(effort.clone());
                    }
                    ReasoningSpec::Thinking {
                        budget_tokens: 2048,
                    }
                } else {
                    ReasoningSpec::Thinking {
                        budget_tokens: 2048,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_spec_none_mode() {
        let config = ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(4096),
            ..Default::default()
        };
        assert!(matches!(
            config.to_spec(ReasoningMode::None),
            ReasoningSpec::None
        ));
    }

    #[test]
    fn to_spec_effort_mode_with_effort() {
        let config = ReasoningConfig {
            effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        };
        assert!(matches!(
            config.to_spec(ReasoningMode::Effort),
            ReasoningSpec::Effort(ReasoningEffort::Medium)
        ));
    }

    #[test]
    fn to_spec_effort_mode_without_effort() {
        let config = ReasoningConfig {
            effort: None,
            budget_tokens: Some(4096),
            ..Default::default()
        };
        assert!(matches!(
            config.to_spec(ReasoningMode::Effort),
            ReasoningSpec::None
        ));
    }

    #[test]
    fn to_spec_thinking_mode() {
        let config = ReasoningConfig {
            budget_tokens: Some(4096),
            ..Default::default()
        };
        assert!(matches!(
            config.to_spec(ReasoningMode::Thinking),
            ReasoningSpec::Thinking {
                budget_tokens: 4096
            }
        ));
    }

    #[test]
    fn to_spec_thinking_mode_default_budget() {
        let config = ReasoningConfig::default();
        assert!(matches!(
            config.to_spec(ReasoningMode::Thinking),
            ReasoningSpec::Thinking {
                budget_tokens: 2048
            }
        ));
    }

    #[test]
    fn to_spec_both_effort_priority() {
        let config = ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            budget_tokens: Some(4096),
            ..Default::default()
        };
        // Both mode should prefer Thinking over Effort,
        // because Thinking is more expressive and Effort can be derived from it.
        // Anthropic only handles Thinking, so preferring Effort would silently drop reasoning.
        assert!(matches!(
            config.to_spec(ReasoningMode::Both),
            ReasoningSpec::Thinking {
                budget_tokens: 4096
            }
        ));
    }

    #[test]
    fn to_spec_both_fallback_to_thinking() {
        let config = ReasoningConfig {
            effort: None,
            budget_tokens: Some(4096),
            ..Default::default()
        };
        assert!(matches!(
            config.to_spec(ReasoningMode::Both),
            ReasoningSpec::Thinking {
                budget_tokens: 4096
            }
        ));
    }

    #[test]
    fn to_spec_both_effort_none_fallback() {
        let config = ReasoningConfig {
            effort: Some(ReasoningEffort::None),
            budget_tokens: Some(4096),
            ..Default::default()
        };
        // Effort::None should be treated as "no effort", fallback to thinking
        assert!(matches!(
            config.to_spec(ReasoningMode::Both),
            ReasoningSpec::Thinking {
                budget_tokens: 4096
            }
        ));
    }

    #[test]
    fn to_spec_enabled_false_disables_reasoning() {
        // ReasoningConfig.enabled=false now disables reasoning.
        let config = ReasoningConfig {
            enabled: Some(false),
            effort: Some(ReasoningEffort::High),
            ..Default::default()
        };
        let spec = config.to_spec(ReasoningMode::Effort);
        assert!(
            matches!(spec, ReasoningSpec::None),
            "enabled=false should disable reasoning"
        );

        let config2 = ReasoningConfig {
            enabled: Some(false),
            budget_tokens: Some(4096),
            ..Default::default()
        };
        let spec2 = config2.to_spec(ReasoningMode::Thinking);
        assert!(
            matches!(spec2, ReasoningSpec::None),
            "enabled=false should disable thinking"
        );
    }

    // ── proptest: ReasoningConfig::to_spec ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn to_spec_never_panics(
                enabled in proptest::option::of(proptest::bool::ANY),
                budget_tokens in proptest::option::of(0u64..100_000),
                effort_idx in 0usize..5,
                mode_idx in 0usize..4,
            ) {
                let effort = match effort_idx {
                    0 => None,
                    1 => Some(ReasoningEffort::None),
                    2 => Some(ReasoningEffort::Low),
                    3 => Some(ReasoningEffort::Medium),
                    4 => Some(ReasoningEffort::High),
                    _ => unreachable!(),
                };
                let mode = match mode_idx {
                    0 => ReasoningMode::None,
                    1 => ReasoningMode::Effort,
                    2 => ReasoningMode::Thinking,
                    3 => ReasoningMode::Both,
                    _ => unreachable!(),
                };
                let config = ReasoningConfig { enabled, budget_tokens, effort };
                let _spec = config.to_spec(mode);
            }

            #[test]
            fn none_mode_always_returns_none(
                enabled in proptest::option::of(proptest::bool::ANY),
                budget_tokens in proptest::option::of(0u64..100_000),
                effort_idx in 0usize..5,
            ) {
                let effort = match effort_idx {
                    0 => None,
                    1 => Some(ReasoningEffort::None),
                    2 => Some(ReasoningEffort::Low),
                    3 => Some(ReasoningEffort::Medium),
                    4 => Some(ReasoningEffort::High),
                    _ => unreachable!(),
                };
                let config = ReasoningConfig { enabled, budget_tokens, effort };
                let spec = config.to_spec(ReasoningMode::None);
                assert!(matches!(spec, ReasoningSpec::None),
                    "Mode::None should always produce Spec::None, got {:?}", spec);
            }

            #[test]
            fn enabled_false_always_returns_none(
                budget_tokens in proptest::option::of(0u64..100_000),
                effort_idx in 0usize..5,
                mode_idx in 0usize..4,
            ) {
                let effort = match effort_idx {
                    0 => None,
                    1 => Some(ReasoningEffort::None),
                    2 => Some(ReasoningEffort::Low),
                    3 => Some(ReasoningEffort::Medium),
                    4 => Some(ReasoningEffort::High),
                    _ => unreachable!(),
                };
                let mode = match mode_idx {
                    0 => ReasoningMode::None,
                    1 => ReasoningMode::Effort,
                    2 => ReasoningMode::Thinking,
                    3 => ReasoningMode::Both,
                    _ => unreachable!(),
                };
                let config = ReasoningConfig { enabled: Some(false), budget_tokens, effort };
                let spec = config.to_spec(mode);
                assert!(matches!(spec, ReasoningSpec::None),
                    "enabled=false should disable reasoning for any mode, got {:?}", spec);
            }
        }
    }
}
