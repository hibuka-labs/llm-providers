//! MiMo model profiles.
//!
//! MiMo supports both OpenAI and Anthropic protocols.
//! OpenAI endpoint does NOT support reasoning_effort (causes 400).
//! Anthropic endpoint supports thinking blocks.

use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_openai_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "mimo",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(128_000),
        },
        // Key: MiMo OpenAI endpoint does NOT support reasoning_effort
        reasoning_mode: ReasoningMode::None,
        supported_extra_params: &[],
    }
}

fn brand_anthropic_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "mimo",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(128_000),
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    }
}

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("mimo@openai", brand_openai_defaults()),
        ("mimo@anthropic", brand_anthropic_defaults()),
        // mimo-v2.5-pro@openai falls back to brand default mimo@openai
    ]
}

pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("mimo-", "mimo")]
}
