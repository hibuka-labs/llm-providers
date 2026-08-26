# Changelog

All notable changes to the llm-providers workspace are documented here.

## [0.1.0] — 2026-08-25

### Added

**llm-trait 0.1.0** — Interface layer (lightweight, no heavy deps):
- `LlmProvider` trait: unified `chat()` / `stream()` / `capabilities()` / `info()` interface
- `RawAdapter` trait: low-level protocol adapter interface
- `LlmConfig` / `LlmBackend` / `Protocol` configuration types
- `ChatRequest` / `ChatResponse` / `ChatStream` / `StreamChunk` request/response types
- `Capabilities` / `ProviderInfo` / `UsageInfo` metadata types
- `LlmError` unified error type
- `ReasoningConfig` / `ReasoningEffort` reasoning support types

**llm-unified 0.1.0** — Implementation layer:
- `AnthropicProtocol`: Anthropic Claude API adapter (`RawAdapter` impl)
- `OpenAiProtocol`: OpenAI Chat Completions API adapter (`RawAdapter` impl)
- `GenericProvider<A>`: wraps any `RawAdapter` into `LlmProvider`
- `MimoProvider`: Mimo AI provider (Anthropic protocol + extensions)
- `DeepSeekProvider`: DeepSeek provider (OpenAI protocol, `supports_thinking`)
- `QwenProvider`: Qwen provider (OpenAI protocol, DashScope endpoint)
- `create_provider()` / `create()` / `from_env()` factory functions
- `llm-cli` binary for CLI verification

### Architecture

- Three-layer design: `llm-trait` (interface) → `llm-unified` (implementation) → `agent-base` (runtime)
- Protocol reuse: OpenAI-compatible providers share `OpenAiProtocol`
- Provider isolation: each provider's extensions are independent
- `agent-base` depends only on `llm-trait`, not `llm-unified`

### Breaking Changes (from agent-base legacy)

- `StreamClient` trait → replaced by `LlmProvider` trait
- `LlmClient` trait → merged into `LlmProvider`
- `LlmProvider` enum → renamed to `LlmBackend`
- `OpenAiAdapter` → replaced by `OpenAiProtocol` + `GenericProvider`
- `AnthropicAdapter` → replaced by `AnthropicProtocol` + `GenericProvider`
- `GenericStreamClient` → replaced by `GenericProvider<A>`

### Test Coverage

- llm-trait: 69 tests (unit + doc-tests)
- llm-unified: 100 tests (unit + wiremock integration)
- Total: 169 tests
