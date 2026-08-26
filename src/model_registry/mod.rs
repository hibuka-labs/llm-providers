//! Model Registry — centralized model knowledge.
//!
//! Each brand has its own module with profile data and brand prefix rules.
//! The `ModelRegistry` collects all data and provides `lookup()` for the factory.

pub mod anthropic;
pub mod deepseek;
pub mod gpt;
pub mod mimo;
pub mod qwen;
pub mod registry;

pub use registry::{ModelProfile, ModelRegistry};
