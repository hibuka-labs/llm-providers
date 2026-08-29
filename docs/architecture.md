# Architecture

This document explains how the pieces fit together and why the layering is the
way it is. For "I just want to add a model", jump to
[adding-a-provider.md](adding-a-provider.md).

## Two crates, one reason

```text
llm-trait      traits + core types        (async-trait, serde, reqwest, thiserror)
llm-unified    adapters + registry + factory + llm-cli
```

The split is not cosmetic. Application and runtime crates depend on `llm-trait`
only, so they can accept an `Arc<dyn LlmProvider>` without pulling in every
vendor adapter, a CLI's argument parser, or the built-in model table. Only the
composition root — the binary that decides *which* provider to build — depends
on `llm-unified`.

The dependency arrow points one way: `llm-unified → llm-trait`. Never the
reverse. `llm-trait` must not know that `llm-unified`, or any specific provider,
exists.

## The three types that matter

### `LlmProvider` — what callers use

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError>;
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream, LlmError>;
    fn capabilities(&self) -> Capabilities;
    fn info(&self) -> ProviderInfo;
}
```

Object-safe on purpose: providers are shared as `Arc<dyn LlmProvider>` across
task boundaries, so a session can hold one without knowing its concrete type.

`capabilities()` is not decorative — it is how calling code decides whether to
send images, whether to request tool calls, and how much context to allow before
summarising.

### `RawAdapter` — what a protocol implements

```rust
#[async_trait]
pub trait RawAdapter: Send + Sync {
    fn build_request(&self, request: &ChatRequest, mode: CallMode) -> Result<RawRequest, LlmError>;
    async fn execute_stream(&self, client: &dyn HttpClient, request: RawRequest) -> Result<ChatStream, LlmError>;
    async fn parse_sse_stream(&self, client: &dyn HttpClient, request: RawRequest, response: HttpResponse) -> Result<ChatStream, LlmError>;
    fn parse_response(&self, body: &[u8]) -> Result<ChatResponse, LlmError>;
    fn capabilities(&self) -> Capabilities;
    fn info(&self) -> ProviderInfo;
    fn supported_modes(&self) -> &[CallMode] { &[CallMode::Stream, CallMode::Once] }
}
```

The division of labour: **the adapter owns the wire format, `GenericProvider`
owns the transport.** The adapter turns a `ChatRequest` into a `RawRequest`
(URL, method, headers, JSON body) and turns bytes back into `StreamChunk`s. It
never opens a connection. `GenericProvider` sends the request, retries it, and
hands the response back to the adapter for parsing.

`parse_sse_stream` exists because of retries: once bytes have started flowing,
a mid-stream failure cannot be retried without duplicating output. So
`GenericProvider` retries only the *initial* HTTP exchange and then calls
`parse_sse_stream` with the already-received response.

Note the asymmetry between the two streaming methods:

- **`parse_sse_stream` is what `GenericProvider` actually calls.** Its default
  returns `Err("parse_sse_stream not implemented")`, so a new adapter must
  override it — streaming appears to work until the first response arrives.
- **`execute_stream` is currently unused by `GenericProvider`.** It is a required
  trait method, so adapters implement it, but nothing in the crate calls it
  (`GenericProvider::execute_stream` is a separate private method). It exists as
  the escape hatch for adapters that want to own the whole send-and-parse flow
  rather than just parsing.

If you are writing an adapter: put your SSE parsing in `parse_sse_stream`. That
is the path your code will run on.

### `HttpClient` — where tests cut in

```rust
#[async_trait]
pub trait HttpClient: Send + Sync {
    async fn send(&self, request: &RawRequest) -> Result<HttpResponse, LlmError>;
}
```

`ReqwestHttpClient` is the production implementation. `HttpResponse` deliberately
hides reqwest types behind a `status` + `Bytes` stream, so a mock can construct
one via `HttpResponse::new` / `HttpResponse::from_text` without spinning up a
server. Use `GenericProvider::with_http_client` for transport-level tests;
`tests/wiremock_*.rs` cover the full stack over real HTTP instead.

## Streaming: `StreamChunk`

```rust
pub enum StreamChunk {
    Text(String),
    Thought(String),
    ThinkingSignature(String),
    ToolCall(Value),
    Usage(UsageInfo),
    Error(String),
    Stop { finish_reason: Option<String> },
}
```

The point of a shared chunk enum is that consumers stop special-casing vendors.
Two consequences are worth knowing:

- **`ToolCall` carries a `Value`, not a typed struct.** Provider tool-call
  payloads differ enough (indices, function vs. name nesting, streaming
  fragments) that adapters normalise what they can and pass the rest through.
  `ChatResponse::tool_calls` *is* typed (`ToolCall { id, name, arguments }`),
  because by then the arguments are complete.
- **`Usage` may arrive more than once, partially filled.** Anthropic sends input
  tokens in `message_start` and output tokens in `message_delta`. Folding events
  with `UsageInfo::merge` (a `None` field never overwrites a known one) is why
  `collect_response` reports real counts rather than zeroes.

`ChatStream` wraps a pinned boxed stream and adds `next()`, `collect_text()`,
and `collect_response()`. Prefer `collect_response()` unless you only want prose
— `collect_text()` discards tool calls and usage.

## Reasoning is three different things

`ReasoningMode` describes *how a provider expresses thinking*, because there is
no common convention:

| Mode | Wire form | Example |
|------|-----------|---------|
| `Effort` | `reasoning_effort: "low" \| "medium" \| "high"` | OpenAI o-series |
| `Thinking` | `thinking: { budget_tokens: N }`, echoed back with a signature | Anthropic |
| `None` | reasoning is not controllable; **never send either field** | many OpenAI-compatible gateways |

`None` is load-bearing. Sending `reasoning_effort` to an endpoint that does not
recognise it is a 400, not an ignored field. That is exactly the class of bug the
registry exists to prevent — a model's *name* does not tell you whether it
accepts a parameter, but its profile does.

Anthropic's thinking `signature` round-trips: multi-turn conversations must send
the signature back with the thinking block or the API rejects the history. It is
surfaced as `StreamChunk::ThinkingSignature` and `ChatResponse::thinking_signature`
for that reason.

## ModelRegistry: knowledge, not guessing

```rust
pub struct ModelProfile {
    pub protocol: Protocol,
    pub provider_name: &'static str,
    pub capabilities: Capabilities,
    pub reasoning_mode: ReasoningMode,
    pub supported_extra_params: &'static [&'static str],
}
```

Profiles are keyed `"<model>@<protocol>"`, so the same model can behave
differently per endpoint — `mimo-v2.5-pro@openai` (no `reasoning_effort`) and
`mimo-v2.5-pro@anthropic` (thinking blocks) are two independent rows.

`lookup()` resolves in this order:

1. **Protocol**: explicit `config.protocol` → URL inference → `OpenAi` default.
2. **Exact** key `model@protocol`.
3. **Brand default** `brand@protocol`, where the brand comes from a prefix table
   (`mimo-` → `mimo`, `gpt-` → `gpt`, …).
4. **Protocol safe default** — conservative capabilities, no per-model claims.

URL inference only auto-selects Anthropic: host is `anthropic.com` (or a
subdomain), or the *path* contains `/anthropic`. Everything else stays on the
`openai` default.

Domain matching respects boundaries on purpose — `notanthropic.com` and
`anthropic.com.evil.com` must not be treated as the official API. That check is
a fuzz target (`domain_matches_fuzz`) for exactly this reason.

### Why not sniff the model name?

Because it breaks the moment someone runs a model behind an unexpected URL:
`gpt-4o` served by a local proxy that rejects `reasoning_effort`; a `claude-*`
alias that only answers on the OpenAI protocol. Name sniffing encodes guesses;
the registry encodes facts, and unknown facts degrade to a safe default instead
of a wrong guess.

## Errors

`LlmError` has four variants — `Config`, `LlmApi { status, message }`, `Llm`,
`Stream` — all `Clone + std::error::Error`, with `status()` to inspect the HTTP
code when the error came from an API response.

`Config` is for problems the caller can fix without a network round trip (missing
key/model/base URL, unsupported protocol). `LlmApi` preserves the provider's
message verbatim; that body is usually the only way to diagnose a 400.

Errors are converted to a consumer's own error type by the caller, not here —
which is why `LlmError` lives in `llm-trait` and depends on nothing project-specific.

## Retries and timeouts

Defaults come from `ProviderConfig`: 15s connect timeout, 120s read timeout,
3 retries, 1s base delay. Retry policy:

- Retried: HTTP 429 and 5xx, on the **initial** exchange only.
- Not retried: 4xx other than 429 (401/403/404 are configuration errors —
  retrying wastes time), and any failure after the stream has started.
- Backoff: `retry_delay * 2^(attempt-1)` plus ≤100ms jitter, capped at 30s.

Once `stream()` returns a `ChatStream`, delivery is the caller's problem: an
in-flight stream that dies surfaces as `StreamChunk::Error` or
`Err(LlmError::Stream)` rather than being transparently restarted, because
restarting would duplicate already-rendered output.

## Non-streaming fallback

`GenericProvider::chat()` checks `supported_modes()`. If the adapter does not
support `CallMode::Once`, it falls back to `stream()` + `collect_response()`.
This is why a new adapter can ship streaming-only and still satisfy `chat()`.

## Testing strategy

| Layer | How |
|-------|-----|
| Type/enum/serde behaviour | unit tests in `llm-trait` (incl. proptest) |
| Adapter request building | unit tests asserting on the produced `RawRequest` JSON |
| Retry/transport logic | `GenericProvider::with_http_client` + a scripted mock client |
| Full request/response cycle | `tests/wiremock_*.rs` against a local mock server |
| Parser robustness | `fuzz/` targets: `parse_openai_response_fuzz`, `parse_anthropic_response_fuzz`, `domain_matches_fuzz` |

Everything but the fuzzers runs offline, which is what makes CI usable — no
secrets, no provider rate limits, no flaky network.

## Known gaps

- **OpenAI Responses API** (`Protocol::OpenAiResponses`) parses from a string but
  has no adapter; `create_provider` returns an explicit error instead of silently
  downgrading. Contributions welcome.
- **Registry coverage** is a small built-in set. Unknown models work, but their
  capabilities are the conservative default, not the real limits.
- **`LlmConfig::options["max_tokens"]`** is read by `from_config` but the factory
  currently resolves `max_tokens` from the profile, so this path is easy to
  misunderstand — see `create_provider_options_max_tokens_ignored`.
