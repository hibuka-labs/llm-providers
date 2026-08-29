# llm-providers

[![crates.io](https://img.shields.io/crates/v/llm-unified.svg)](https://crates.io/crates/llm-unified)
[![Documentation](https://docs.rs/llm-unified/badge.svg)](https://docs.rs/llm-unified)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![coverage](https://img.shields.io/badge/coverage-94%25-green)](CONTRIBUTING.md#coverage)
[![CI](https://github.com/hibuka-labs/llm-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/hibuka-labs/llm-providers/actions/workflows/ci.yml)

One trait, many LLM backends. A unified Rust provider layer for
OpenAI-compatible and Anthropic-compatible APIs — streaming, tool calls,
reasoning traces, and usage in a single shape.

If you have ever written `match provider { OpenAi => …, Anthropic => …, DeepSeek => … }`
and parsed SSE frames three different ways, this is for you.

## Why

Provider APIs differ in ways that have nothing to do with model quality:

- Streaming frames are `data: {…}` in one API and `event: content_block_delta` in another.
- Tool-call arguments stream as *incremental fragments* in OpenAI but as *one whole object* in Anthropic.
- "Reasoning" is `reasoning_effort`, `thinking.budget_tokens`, or a field that gets you an HTTP 400.
- Usage arrives split across several stream events, so naively overwriting it zeroes out counts.

`llm-providers` normalises all of that behind one object-safe trait, so
application code never learns which vendor it is talking to.

## Features

- **Single trait API** — `chat()`, `stream()`, `capabilities()`, `info()`; object-safe, so `Arc<dyn LlmProvider>` just works.
- **Two protocol adapters cover dozens of endpoints** — any OpenAI- or Anthropic-compatible base URL works without new code.
- **Streaming and non-streaming** from the same request type, with `collect_text()` / `collect_response()` helpers.
- **Incremental tool-call assembly** across stream chunks, including truncated or invalid argument payloads.
- **Reasoning unified** — `Effort` / `Thinking` / `None` modes mapped per provider.
- **Model registry** — capability metadata keyed by `model@protocol`, with brand-prefix and URL fallbacks.
- **Injectable HTTP client** — the `HttpClient` trait lets you mock transport in tests, no network required.
- **No provider lock-in** — implement `RawAdapter` to add a wire protocol.

## Crates

| Crate | Version | Description |
|-------|---------|-------------|
| `llm-trait` | 0.1.0 | Traits and core types. No heavy deps — safe for library authors to depend on. |
| `llm-unified` | 0.1.0 | Protocol adapters, model registry, factory, and the `llm-cli` binary. |

`llm-unified` depends on `llm-trait`. Code that only *consumes* a provider can
depend on `llm-trait` alone and stay decoupled from concrete implementations.

## Install

```toml
[dependencies]
llm-trait = "0.1"
llm-unified = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Requires Rust 1.88+ (edition 2024).

## Quick Start

```rust
use llm_trait::{ChatMessage, ChatRequest, LlmConfig, StreamChunk};
use llm_unified::create_provider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure: API key + model + base URL
    let config = LlmConfig {
        protocol: None, // inferred from base_url and the model registry
        api_key: std::env::var("LLM_API_KEY")?,
        model: "gpt-4o-mini".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        options: Default::default(),
    };

    // 2. Build the provider (protocol + capabilities resolved by the registry)
    let provider = create_provider(&config)?;

    let request = ChatRequest::new(vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Reply with one word."),
    ]);

    // 3a. Non-streaming
    let response = provider.chat(request.clone()).await?;
    println!("{}", response.content);
    println!("finish: {:?}, usage: {:?}", response.finish_reason, response.usage);

    // 3b. Streaming
    let mut stream = provider.stream(request).await?;
    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk::Text(t) => print!("{t}"),
            StreamChunk::Thought(t) => eprintln!("[thinking] {t}"),
            StreamChunk::ToolCall(call) => eprintln!("[tool] {call}"),
            StreamChunk::Usage(usage) => eprintln!("[usage] {usage:?}"),
            StreamChunk::Stop { finish_reason } => {
                eprintln!("\n[stop] {finish_reason:?}");
                break;
            }
            StreamChunk::Error(e) => eprintln!("[error] {e}"),
            StreamChunk::ThinkingSignature(_) => {}
        }
    }
    Ok(())
}
```

Prefer the stream to be collapsed for you? `collect_text` and `collect_response`
consume the stream, so pick one per stream:

```rust
let stream = provider.stream(request).await?;
let text = stream.collect_text().await?;         // prose only

let full = provider.stream(request).await?.collect_response().await?; // + tool calls + usage
```

`base_url` is required — there is no implicit default endpoint. To configure
nothing in code at all:

```rust
use llm_unified::from_env;
let provider = from_env()?; // LLM_API_KEY / LLM_MODEL / LLM_BASE_URL / LLM_PROTOCOL
```

## Supported Providers

Built into the registry:

| Provider | Protocol | Example models |
|----------|----------|----------------|
| OpenAI | OpenAI | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | Anthropic | `claude-sonnet-4-20250514` |
| DeepSeek | OpenAI | `deepseek-chat`, `deepseek-reasoner` |
| Qwen (DashScope) | OpenAI | `qwen-plus`, `qwen-max`, `qwen-vl` |
| MiMo | OpenAI / Anthropic | `mimo-v2.5-pro` |

Anything else works too — unrecognised models fall back to the protocol's safe
default profile:

| Endpoint type | How to use it |
|---------------|---------------|
| OpenAI-compatible (vLLM, Ollama, LM Studio, OpenRouter, one-api, Azure, …) | set `base_url`; protocol inferred or `protocol: Some(Protocol::OpenAi)` |
| Anthropic-compatible (`…/anthropic`, Bedrock proxies, gateways, …) | set `base_url` — a path containing `/anthropic` auto-selects the Anthropic adapter |
| Local proxies | point `base_url` at `http://localhost:…` |

Protocol resolution order is **explicit `protocol` → URL inference → `openai` default**.

## Environment Variables

| Variable | Required | Meaning |
|----------|----------|---------|
| `LLM_API_KEY` | yes | Provider API key |
| `LLM_MODEL` | yes | Model identifier |
| `LLM_BASE_URL` | yes | Endpoint base URL |
| `LLM_PROTOCOL` | no | Force `openai` or `anthropic` |

Copy `.env.example` to `.env` and `source` it. `.env` is git-ignored.

## CLI

Smoke-test an endpoint without writing Rust:

```bash
export LLM_API_KEY=sk-...
cargo run --bin llm-cli -- \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --message "Hello" \
    --stream
```

```text
--model <MODEL>        Model name         [env: LLM_MODEL]
--api-key <API_KEY>    API key            [env: LLM_API_KEY]
--base-url <BASE_URL>  Base URL           [env: LLM_BASE_URL]
--protocol <PROTOCOL>  openai | anthropic [env: LLM_PROTOCOL]
--message <MESSAGE>    User message
--system <SYSTEM>      System prompt
--stream               Stream tokens as they arrive
--tools <TOOLS>        Path to a JSON file of tool definitions
--max-tokens <N>       Max output tokens  [default: 4096]
```

## Architecture

```text
        your app / agent runtime / llm-cli
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │  llm-unified (implementation)              │
   │    factory::create_provider / from_env     │
   │    ModelRegistry  → model@protocol profile │
   │    GenericProvider<A: RawAdapter>          │
   │      ├── OpenAiProtocol                    │
   │      └── AnthropicProtocol                 │
   └───────────────────┬────────────────────────┘
                       ▼
   ┌────────────────────────────────────────────┐
   │  llm-trait (interface; no heavy deps)      │
   │    LlmProvider, RawAdapter, HttpClient     │
   │    ChatRequest / ChatResponse / ChatStream │
   │    LlmConfig, Capabilities, UsageInfo      │
   └────────────────────────────────────────────┘
```

Two layers matter in practice:

1. **`RawAdapter`** owns the wire format: build the HTTP request, parse SSE,
   assemble tool calls. **`GenericProvider`** owns transport: retries, timeouts,
   and turning a `RawAdapter` into an `LlmProvider`.
2. **`ModelRegistry`** owns *knowledge* — which model speaks which protocol,
   whether it accepts `reasoning_effort`, and its token ceilings. This is what
   stops the code from guessing based on model-name strings.

## Adding a Provider

**OpenAI- or Anthropic-compatible?** No new code needed — pass `base_url`
(optionally `protocol`). Add a registry entry only if you want accurate
capabilities:

```rust
// src/model_registry/mybrand.rs
pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![("mybrand@openai", brand_openai_defaults())]
}

pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("my-", "mybrand")]
}
```

Then register both in `ModelRegistry::builtin()`.

**A new wire protocol?** Implement `RawAdapter` (`build_request`,
`execute_stream`, `parse_sse_stream`, `parse_response`, `capabilities`, `info`)
and wrap it:

```rust
use llm_unified::GenericProvider;
let provider = GenericProvider::new(Box::new(MyProtocol::from_config(&config)));
```

A full checklist is in [docs/adding-a-provider.md](docs/adding-a-provider.md).

## Testing

Everything runs offline against `wiremock`:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo llvm-cov --workspace --ignore-filename-regex 'bin/llm-cli.rs'   # ~94%
```

Fuzz targets (needs nightly + `cargo install cargo-fuzz`):

```bash
cd fuzz
cargo +nightly fuzz run parse_openai_response_fuzz
cargo +nightly fuzz run parse_anthropic_response_fuzz
cargo +nightly fuzz run domain_matches_fuzz
```

CI fuzzes all three on pull requests, nightly, on a weekly long run, and on
demand — see [fuzzing in CI](#fuzzing-in-ci) below.

## Fuzzing in CI

`.github/workflows/fuzz.yml` runs the real fuzzers (nightly + libFuzzer), not
just a build check. The cost is a single knob — `total_seconds`, split evenly
over the three targets:

| Trigger | Total fuzzing | Job timeout | Fails the build? |
|---------|---------------|-------------|------------------|
| Pull request | 60s (~20s/target) | 30 min | no (advisory) |
| Weekly schedule (Sun 03:00 UTC) | 900s (~300s/target) | 60 min | yes |
| Manual dispatch | you choose | 60 min | yes |

A measured 30s dispatch run took **5m10s** wall-clock: 38s to compile
cargo-fuzz (cached afterwards), 3m37s for the ASan build, ~33s fuzzing, the rest
setup. So the floor is the **build**, not the fuzzing — that run still managed
5.5M executions (63k–380k execs/s per target) in its 10s slices. PR runs report
fuzzing as an advisory warning so a nightly hiccup can't block a merge; the
weekly schedule is the hard gate. To fuzz more, go to **Actions → fuzz → Run
workflow** and set `total_seconds`.

Seed corpora live in `fuzz/seeds/<target>/` and are committed — every run starts
from realistic JSON shapes (valid responses, error bodies, truncated tool-call
arguments, domain-matching traps) instead of random noise. Add a seed file when
you fix a parser bug: that is the cheapest permanent regression guard. Generated
input stays ignored (`fuzz/corpus/`, `fuzz/artifacts/`).

On a crash, CI uploads `fuzz/artifacts/` and the log shows the stack trace.
Reproduce with:

```bash
cd fuzz && cargo +nightly fuzz run <target> artifacts/<target>/<file>
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Layering, trait contracts, streaming/tool-call semantics, registry design, retry policy |
| [docs/adding-a-provider.md](docs/adding-a-provider.md) | Three levels of provider integration, with code and test requirements |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup and PR expectations |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |

API reference: `cargo doc --open`.

## Contributing

Contributions are welcome. New providers, protocol fixes, and regression tests
are the most useful things you can send. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first, and open an issue before large
refactors.

## Stability

This project is in early development (v0.1.x). The `LlmProvider` and `RawAdapter`
traits are usable but not yet frozen — expect minor signature changes as the
ecosystem settles. Breaking changes are called out in
[CHANGELOG.md](CHANGELOG.md) and shipped only in minor versions.

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work shall be licensed as above, without any
additional terms or conditions.
