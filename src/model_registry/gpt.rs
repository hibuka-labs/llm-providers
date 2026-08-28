//! OpenAI GPT model profiles.

use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "openai",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: false,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
        },
        reasoning_mode: ReasoningMode::None,
        supported_extra_params: &[],
    }
}

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("gpt@openai", brand_defaults()),
        (
            "gpt-4o@openai",
            ModelProfile {
                capabilities: Capabilities {
                    max_output_tokens: Some(16_384),
                    ..brand_defaults().capabilities
                },
                ..brand_defaults()
            },
        ),
        (
            "gpt-4o-mini@openai",
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
    vec![("gpt-", "gpt")]
}
