//! # llm-trait
//!
//! Trait definitions and core types for LLM providers.
//!
//! This crate is the **interface layer**: the traits and types that both
//! `llm-unified` (implementations) and consumer applications (runtimes, CLIs,
//! servers) depend on. Depending on this crate alone is enough to accept and
//! call a provider without pulling in any vendor adapter, the model registry,
//! or the CLI.
//!
//! It is not dependency-free: [`ReqwestHttpClient`] is the default transport, so
//! reqwest and its TLS stack come along. That trade-off is what lets adapters be
//! unit-tested against a mocked [`HttpClient`] without a second crate in the
//! dependency graph.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────┐    ┌──────────────────────────┐
//! │     llm-unified          │    │     your app             │
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
pub mod http_client;
pub mod message;
pub mod provider;
pub mod raw_adapter;
pub mod reasoning;
pub mod request;
pub mod response;
pub mod types;

// Re-export key types at crate root for convenience.
pub use backend::Protocol;
pub use capabilities::{Capabilities, ProviderInfo};
pub use config::LlmConfig;
pub use error::LlmError;
pub use http_client::{HttpClient, HttpResponse, ReqwestHttpClient};
pub use message::{ChatMessage, ImageAttachment, ImageDetail, ToolCallMessage};
pub use provider::LlmProvider;
pub use raw_adapter::{CallMode, HttpMethod, RawAdapter, RawRequest, StreamState};
pub use reasoning::{ReasoningConfig, ReasoningEffort, ReasoningMode, ReasoningSpec};
pub use request::{ChatRequest, ResponseFormat};
pub use response::{
    ChatResponse, ChatStream, FinishReason, StreamChunk, ToolCall, extract_tool_calls,
    parse_finish_reason,
};
pub use types::UsageInfo;
