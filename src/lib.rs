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
//!     base_url: "https://api.example-mimo.com/v1".to_string(),
//!     options: std::collections::HashMap::new(),
//! };
//! let provider = create_provider(&config)?;
//! # Ok(())
//! # }
//! ```

pub mod factory;
pub mod generic;
pub mod model_registry;
pub mod protocol;

pub use factory::{create, create_provider, from_env};
pub use generic::{GenericProvider, ProfiledProvider};
pub use protocol::anthropic::AnthropicProtocol;
pub use protocol::openai::OpenAiProtocol;

#[cfg(feature = "fuzzing")]
pub mod fuzz {
    pub use super::model_registry::registry::fuzz_exports as model_registry;
    pub use super::protocol::anthropic::fuzz_exports as anthropic;
    pub use super::protocol::openai::fuzz_exports as openai;
}
