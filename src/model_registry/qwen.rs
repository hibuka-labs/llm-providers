//! Qwen (Tongyi Qianwen) model profiles.

use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "qwen",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(8_192),
        },
        reasoning_mode: ReasoningMode::None,
        supported_extra_params: &[],
    }
}

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("qwen@openai", brand_defaults()),
        ("qwen-plus@openai", brand_defaults()),
        (
            "qwen-max@openai",
            ModelProfile {
                capabilities: Capabilities {
                    max_output_tokens: Some(32_768),
                    ..brand_defaults().capabilities
                },
                ..brand_defaults()
            },
        ),
        (
            "qwen-vl@openai",
            ModelProfile {
                capabilities: Capabilities {
                    supports_vision: true,
                    ..brand_defaults().capabilities
                },
                ..brand_defaults()
            },
        ),
    ]
}

pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("qwen-", "qwen")]
}
