//! DeepSeek model profiles.

use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "deepseek",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
        },
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &["reasoning_effort"],
    }
}

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("deepseek@openai", brand_defaults()),
        ("deepseek-chat@openai", brand_defaults()),
        (
            "deepseek-reasoner@openai",
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
    vec![("deepseek-", "deepseek")]
}
