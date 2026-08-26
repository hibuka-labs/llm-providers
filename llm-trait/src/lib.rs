//! # llm-trait
//!
//! Trait definitions and core types for LLM providers.
//!
//! This crate is the **interface layer** — extremely lightweight, with no heavy
//! dependencies. It defines the traits and types that both `llm-unified` (implementation)
//! and `agent-base` (runtime) depend on.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────┐    ┌──────────────────────────┐
//! │     llm-unified          │    │     agent-base           │
//! │     (implementation)     │    │     (runtime)            │
//! └──────────┬───────────────┘    └──────────┬───────────────┘
//!            │                               │
//!            │  depends on                   │  depends on
//!            ▼                               ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │                    llm-trait (this crate)                 │
//! │  LlmProvider trait, RawAdapter trait, core types         │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```
//! use llm_trait::{LlmConfig, LlmProvider, ChatRequest, ChatMessage};
//! ```

pub mod backend;
pub mod capabilities;
pub mod config;
pub mod error;
pub mod message;
pub mod provider;
pub mod raw_adapter;
pub mod reasoning;
pub mod request;
pub mod response;
pub mod types;

// Re-export key types at crate root for convenience.
pub use backend::{LlmBackend, Protocol};
pub use capabilities::{Capabilities, ProviderInfo};
pub use config::LlmConfig;
pub use error::LlmError;
pub use message::{ChatMessage, ImageAttachment, ImageDetail, ToolCallMessage};
pub use provider::LlmProvider;
pub use raw_adapter::{CallMode, HttpMethod, RawAdapter, RawRequest, StreamState};
pub use reasoning::{ReasoningConfig, ReasoningEffort};
pub use request::{ChatRequest, ResponseFormat};
pub use response::{
    ChatResponse, ChatStream, FinishReason, StreamChunk, ToolCall,
    extract_tool_calls, parse_finish_reason,
};
pub use types::UsageInfo;
