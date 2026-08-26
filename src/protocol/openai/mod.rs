//! OpenAI Chat Completions protocol implementation.
//!
//! Implements `RawAdapter` for the OpenAI Chat Completions API.
//! Also serves as the base protocol for OpenAI-compatible providers
//! (DeepSeek, Qwen, Azure, local models, etc.).

mod protocol;

pub use protocol::OpenAiProtocol;
