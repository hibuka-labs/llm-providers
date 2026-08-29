# Changelog

All notable changes to the `llm-providers` workspace are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LlmError::status()` — inspect the HTTP status code carried by an `LlmApi` error.
- `GenericProvider::with_http_client()` — inject a custom `HttpClient` to stub
  transport in tests or wrap requests in custom middleware.
- `impl FromStr for FinishReason` — string parsing via `"length".parse()`,
  alongside the existing inherent `FinishReason::from_str`.
- Unit tests covering retry behaviour through an injected mock HTTP client
  (5xx retried, 4xx not retried).
- `examples/quickstart.rs`, runnable against a real endpoint.
- MIT `LICENSE`; crate metadata (`license`, `repository`, `description`,
  `keywords`, `categories`, `rust-version`) for both crates.
- Documentation set: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `docs/architecture.md`, `docs/adding-a-provider.md`.
- CI (fmt, clippy with `-D warnings`, build + test on stable/beta/MSRV, docs,
  fuzz build, llvm-cov coverage) and Dependabot for Cargo and GitHub Actions.
- `codecov.yml` with an 85% project / 80% patch threshold, excluding
  `tests/`, `examples/`, `fuzz/` and the `llm-cli` binary.
- 27 new unit tests covering previously untested public API: `LlmConfig::from_env`
  (all required-variable branches), `HttpResponse` text/stream semantics, the
  credential-redacting `Debug` impls, `ResponseFormat::to_api_value`,
  `FinishReason` string conversions, and `ProtocolParseError` display.
  Workspace coverage is now 94.1% regions / 93.9% lines (284 tests, CLI binary
  excluded; was ~92.6% regions before these tests).

### Changed
- **Breaking:** `Debug` for `LlmConfig` and `RawRequest` no longer prints
  credentials — `api_key` renders as `<redacted>` and authentication-related
  headers are redacted. This prevents accidental key leakage in logs and panics.
- `LlmConfig::default()` is now a derived `Default` impl.
- Internal audit identifiers (`P0-1`, `P2-13`, …) removed from code comments and
  assertion messages in favour of plain-language explanations.
- Provider-specific example endpoints in tests and doc comments replaced with
  neutral placeholders.
- `llm-unified` now declares `llm-trait = { version = "0.1" }`, required to
  publish both crates to crates.io.

### Fixed
- `max_tokens_with_profile_uses_profile_value` asserted a stale MiMo
  `max_output_tokens` value (8192) after the registry moved to 128K. The test now
  checks the registry value *and* that it reaches the wire request.

## [0.1.0] — 2026-08-25

### Added

**`llm-trait` 0.1.0** — interface layer:
- `LlmProvider` trait: unified `chat()` / `stream()` / `capabilities()` / `info()`
- `RawAdapter` trait: low-level protocol adapter interface
- `HttpClient` trait + `ReqwestHttpClient`, with `HttpResponse` hiding reqwest types
- `LlmConfig` / `Protocol` configuration types
- `ChatRequest` / `ChatResponse` / `ChatStream` / `StreamChunk` request & response types
- `ChatMessage` with system/user/assistant/tool/custom variants, images, and
  ephemeral messages
- `Capabilities` / `ProviderInfo` / `UsageInfo` metadata types
- `LlmError` unified error type
- `ReasoningConfig` / `ReasoningEffort` / `ReasoningMode` / `ReasoningSpec`

**`llm-unified` 0.1.0** — implementation layer:
- `OpenAiProtocol`: OpenAI Chat Completions adapter
- `AnthropicProtocol`: Anthropic Messages adapter
- `GenericProvider`: turns any `RawAdapter` into an `LlmProvider`, with retries
  and exponential backoff
- `ProfiledProvider`: overrides `capabilities()` / `info()` from a `ModelProfile`
- `ModelRegistry`: `model@protocol` profiles with brand-prefix and URL fallbacks
- `create_provider()` / `create()` / `from_env()` factory functions
- `llm-cli` binary for endpoint verification
- Fuzz targets for both response parsers and URL domain matching

### Architecture

- Two-crate split: consumers depend on `llm-trait` only; `llm-unified` holds
  implementations
- Protocol reuse: all OpenAI-compatible endpoints share `OpenAiProtocol`
- Model knowledge centralised in the registry instead of scattered per-provider code

### Breaking changes (relative to the previous in-tree client)

- `StreamClient` trait → replaced by `LlmProvider`
- `LlmClient` trait → merged into `LlmProvider`
- `LlmProvider` enum → replaced by the `Protocol` enum
- `OpenAiAdapter` / `AnthropicAdapter` → `OpenAiProtocol` / `AnthropicProtocol`
  wrapped by `GenericProvider`
- `GenericStreamClient` → `GenericProvider`

[Unreleased]: https://github.com/hibuka-labs/llm-providers/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hibuka-labs/llm-providers/releases/tag/v0.1.0
