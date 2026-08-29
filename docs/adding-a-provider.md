# Adding a Provider or Model

There are three levels of work, in increasing order of effort. Start at the top —
most requests do not need level 3.

| Situation | What you need |
|-----------|---------------|
| Endpoint speaks OpenAI or Anthropic, defaults are fine | **nothing** — just set `base_url` |
| Endpoint works but has facts worth recording (token limits, `reasoning_effort` support) | level 2: a registry entry |
| Endpoint speaks a genuinely different wire protocol | level 3: a `RawAdapter` |

---

## Level 1 — Use an existing protocol (no code)

```rust
use llm_trait::{LlmConfig, Protocol};
use llm_unified::create_provider;

// Any OpenAI-compatible server: vLLM, Ollama, LM Studio, OpenRouter, one-api, …
let config = LlmConfig {
    protocol: Some(Protocol::OpenAi),
    api_key: "…".into(),
    model: "qwen2.5-coder:32b".into(),
    base_url: "http://localhost:11434/v1".into(),
    options: Default::default(),
};
let provider = create_provider(&config)?;
```

Protocol resolution order is **explicit `protocol` → URL inference → `openai`
default**. URL inference currently auto-selects Anthropic when the host is
`anthropic.com` (or a subdomain) or the path contains `/anthropic`. Otherwise,
being explicit is one line and removes all guesswork.

Verify with the CLI before writing anything:

```bash
cargo run --bin llm-cli -- --model <model> --base-url <url> --api-key <key> --message "hi" --stream
```

---

## Level 2 — Add a model profile

Add this when the model's capabilities differ from the protocol default, or when
callers need accurate numbers from `capabilities()`.

### 1. Create the module

```rust
// src/model_registry/mybrand.rs
use super::ModelProfile;
use llm_trait::{Capabilities, Protocol, ReasoningMode};

fn brand_openai_defaults() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "mybrand",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(8_192),
        },
        // Pick deliberately: see the table below.
        reasoning_mode: ReasoningMode::None,
        supported_extra_params: &[],
    }
}

/// Exact-match keys, format `"<model>@<protocol>"`.
pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![
        ("mybrand@openai", brand_openai_defaults()),
        ("my-Flagship-1@openai", ModelProfile { ..brand_openai_defaults() }),
    ]
}

/// Prefix-match fallback so unknown `my-*` models still resolve to the brand profile.
pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("my-", "mybrand")]
}
```

### 2. Register it

```rust
// src/model_registry/mod.rs
mod mybrand;

// src/model_registry/registry.rs, in ModelRegistry::builtin()
for (name, profile) in super::mybrand::profiles() {
    profiles.insert(name.to_string(), profile);
}
brand_prefixes.extend(super::mybrand::brand_prefixes());
```

### 3. Choose `reasoning_mode` correctly

This is the field that causes real outages. Getting it wrong is not cosmetic:

| Mode | Request body effect | Use when |
|------|---------------------|----------|
| `None` | nothing reasoning-related is sent | endpoint rejects unknown reasoning fields (→ HTTP 400), or has no reasoning control at all |
| `Effort` | `"reasoning_effort": "low\|medium\|high"` | OpenAI-style reasoning models |
| `Thinking` | `"thinking": { "budget_tokens": N }` | Anthropic-style thinking blocks |

If you are unsure, test with `Effort` first and check for a 400 — but `None` is
the safe default, since the cost of omitting a supported field is lower than the
cost of sending an unsupported one.

### 4. Test it

At minimum, cover exact match, brand-prefix fallback, and protocol inference:

```rust
#[test]
fn registry_resolves_mybrand() {
    let registry = ModelRegistry::builtin();

    let exact = registry.lookup("my-Flagship-1", Some("https://api.mybrand.com/v1"), None);
    assert_eq!(exact.provider_name, "mybrand");

    // unknown model with known prefix → brand default
    let brand = registry.lookup("my-cheap-7b", Some("https://api.mybrand.com/v1"), None);
    assert_eq!(brand.provider_name, "mybrand");
    assert_eq!(brand.reasoning_mode, ReasoningMode::None);
}
```

Plus a wire-level test that the generated request body looks right
(`src/protocol/openai/protocol.rs` `mod tests` has `with_model_profile` helpers),
and a wiremock round trip in `tests/` if the provider has a quirk worth locking in.

---

## Level 3 — Implement a new wire protocol

Create `src/protocol/<name>/` with `mod.rs`, `protocol.rs`, and optionally
`types.rs` for `serde` structs. Implement `RawAdapter`:

```rust
#[async_trait]
impl RawAdapter for MyProtocol {
    fn build_request(&self, request: &ChatRequest, mode: CallMode) -> Result<RawRequest, LlmError> {
        Ok(RawRequest {
            url: format!("{}/chat", self.base_url),
            method: HttpMethod::Post,
            headers: HashMap::from([
                ("authorization".to_string(), format!("Bearer {}", self.api_key)),
            ]),
            body: serde_json::json!({ /* … */ }),
            stream: matches!(mode, CallMode::Stream),
        })
    }

    async fn parse_sse_stream(
        &self,
        _client: &dyn HttpClient,
        _request: RawRequest,
        response: HttpResponse,
    ) -> Result<ChatStream, LlmError> {
        // Decode SSE frames -> your event enum -> Vec/Stream of StreamChunk.
        Ok(ChatStream::new(Box::pin(stream)))
    }

    fn parse_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError> { /* … */ }
    fn capabilities(&self) -> Capabilities { /* … */ }
    fn info(&self) -> ProviderInfo { /* … */ }
}
```

Notes that save time:

- **Do not send HTTP yourself.** Return a `RawRequest`; `GenericProvider` handles
  retries and timeouts. Implement `parse_sse_stream` (not just `execute_stream`)
  so retries behave correctly.
- **Stream tool-call fragments need assembling.** Keep per-index buffers in your
  adapter's stream state and emit a complete `ToolCall` only when the arguments
  are finished — `extract_tool_calls` in `llm-trait/src/response.rs` and the
  delta-merging logic in `ChatStream::collect_response` show the pattern
  OpenAI-style indexed deltas need.
- **Merge usage rather than overwriting it** with `UsageInfo::merge`.
- **Parse untrusted JSON defensively.** `serde_json` errors are fine; panics are
  not. No `unwrap()` on values that came off the wire — unknown fields, null
  where you expect a string, and truncated payloads all must resolve to
  `LlmError`.
- Register the protocol in `src/factory.rs` `build_protocol()` if it should be
  reachable from `create_provider`, and add a `Protocol` variant in
  `llm-trait/src/backend.rs` plus its `FromStr`/`as_str` arms.

### Required tests for a new adapter

1. `build_request` for both `CallMode::Once` and `CallMode::Stream`.
2. Happy-path streaming, including a mid-stream `Stop`.
3. Tool-call arguments split across several SSE frames.
4. Malformed / unknown / truncated frames → `Err`, never panic.
5. Error status bodies (`400` with provider JSON, `401`, `429`) map to
   `LlmError::LlmApi` with the message preserved.
6. A `tests/` wiremock file exercising the whole path through `GenericProvider`.

---

## Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `cargo fmt --all` applied
- [ ] `cargo doc --no-deps` has no broken links
- [ ] New public items documented
- [ ] Registry entry (if any) has exact-match, prefix, and inference tests
- [ ] A bug fix ships with a regression test
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] No API keys, internal hostnames, or customer data in code, tests, or docs
