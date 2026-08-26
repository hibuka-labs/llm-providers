//! # llm-unified
//!
//! Unified LLM provider implementations.
//!
//! This crate provides concrete implementations of `llm_trait::LlmProvider`
//! for various LLM backends. It depends on `llm-trait` for trait definitions
//! and core types.
//!
//! ## Quick Start
//!
//! ```no_run
//! use llm_trait::LlmConfig;
//! use llm_unified::create_provider;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LlmConfig {
//!     protocol: None,
//!     api_key: "tp-xxx".to_string(),
//!     model: "mimo-v2.5-pro".to_string(),
//!     base_url: "https://token-plan-cn.xiaomimimo.com/v1".to_string(),
//!     options: std::collections::HashMap::new(),
//! };
//! let provider = create_provider(&config)?;
//! # Ok(())
//! # }
//! ```

pub mod protocol;
pub mod providers;
pub mod generic;
pub mod factory;

pub use factory::{create_provider, create, from_env};
pub use generic::GenericProvider;
pub use providers::mimo::MimoProvider;
pub use providers::deepseek::DeepSeekProvider;
pub use providers::qwen::QwenProvider;
pub use protocol::anthropic::AnthropicProtocol;
pub use protocol::openai::OpenAiProtocol;
