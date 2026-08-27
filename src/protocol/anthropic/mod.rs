//! Anthropic protocol implementation.
//!
//! Implements `RawAdapter` for the Anthropic Messages API.
//! Uses `eventsource-stream` for SSE parsing,
//! with `anthropic-rs-api` types for request/response structures.

mod protocol;
mod types;

pub use protocol::AnthropicProtocol;

#[cfg(feature = "fuzzing")]
pub use protocol::fuzz_exports;
