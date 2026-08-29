//! Anthropic Claude model profiles.

use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "anthropic",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    }
}

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("anthropic@anthropic", brand_defaults()),
        ("claude@anthropic", brand_defaults()),
        (
            "claude-sonnet-4-20250514@anthropic",
            ModelProfile {
                capabilities: Capabilities {
                    max_output_tokens: Some(16_384),
                    ..brand_defaults().capabilities
                },
                ..brand_defaults()
            },
        ),
    ]
}

pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("claude-", "claude"), ("anthropic-", "anthropic")]
}
