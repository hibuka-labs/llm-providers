# Contributing to llm-providers

Thanks for taking the time to contribute. This guide covers setup, conventions,
and what a review-ready pull request looks like.

## Code of Conduct

By participating you agree to keep discussions constructive. Be kind; assume
good faith. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Getting Started

Requirements:

- Rust 1.88 or newer (the workspace uses edition 2024). `rustup update stable`.
- Git.
- Optional, for fuzzing: nightly toolchain and `cargo install cargo-fuzz`.

```bash
git clone https://github.com/hibuka-labs/llm-providers.git
cd llm-providers
cargo build
cargo test
```

All tests run offline — they use `wiremock` to stand in for provider APIs. No
API keys are needed to build, test, or lint.

## Project Layout

```text
llm-trait/            interface layer: traits + core types (keep deps minimal)
src/factory.rs        create_provider / create / from_env
src/generic.rs        GenericProvider<RawAdapter> → LlmProvider, retries
src/protocol/         wire adapters: openai/, anthropic/
src/model_registry/   per-brand model capability data
tests/                wiremock integration tests
examples/             runnable examples (cargo run --example quickstart)
fuzz/                 cargo-fuzz targets (separate crate, not in the default build path)
docs/                 design documents
```

## Coverage

Measured with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
(branch coverage, not line-only heuristics):

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov

# Terminal summary (excludes the llm-cli binary — see codecov.yml)
cargo llvm-cov --workspace --summary-only --ignore-filename-regex 'bin/llm-cli.rs'

# which lines are uncovered
cargo llvm-cov --workspace --ignore-filename-regex 'bin/llm-cli.rs' --show-missing-lines

# browsable annotated source, written to target/llvm-cov/html
cargo llvm-cov --workspace --html

# lcov file — exactly what the CI coverage job uploads
cargo llvm-cov --workspace --lcov --output-path lcov.info

# include doc examples — needs nightly (rustdoc -Z persist-doctests)
cargo +nightly llvm-cov --workspace --doctests

# with the `fuzzing` feature on. Adds no coverable regions (those modules are
# pure `pub use` re-exports) but proves the cfg path compiles and runs.
cargo llvm-cov --workspace --features fuzzing
```

Current status (stable toolchain, 284 tests, `src/bin/llm-cli.rs` excluded):

| Scope | Regions | Lines | Functions |
|-------|---------|-------|-----------|
| **workspace total** | **94.1%** | **93.9%** | **94.2%** |
| `llm-trait` | 95.9% | 95.6% | 94.3% |
| `llm-unified` | 92.6% | 91.9% | 91.8% |

Within `llm-unified`, the model registry and per-brand profile modules are at
≥97%; the protocol adapters sit near 92% because their remaining gaps are error
and transport branches that need a genuinely broken socket to reach.

The `Branches` column in the report reads `-`: rustc emits branch instrumentation
only on nightly. **Region** coverage is the number to trust here — it is finer than
line coverage, counting each match arm and `if`/`else` path separately. Adding
`--doctests` on nightly lowers the totals slightly (~92.5%), because doctests are
not instrumented under the same cfg as the library.

`codecov.yml` sets an 85% project threshold with a 2% tolerance, and an 80%
patch target — the CI `coverage` job uploads `lcov.info` when a `CODECOV_TOKEN`
secret is configured, and skips silently otherwise.

If a local number looks suddenly *worse* or *better* than expected, stale
instrumented artifacts are the usual cause: runs with `--html`, `--doctests` or
`--features fuzzing` leave data in `target/llvm-cov-target` that gets merged into
the next report. Reset before trusting an outlier:

```bash
rm -rf target/llvm-cov-target target/llvm-cov
```

The uncovered remainder is mostly deliberate: reqwest transport plumbing in
`ReqwestHttpClient::send`, `reqwest` client construction, and error branches
that need a genuinely broken socket. If you touch a parser, add a case rather
than accepting the drop — fuzz corpora under `fuzz/` are good sources of inputs.

## Making Changes

1. **Start with an issue.** For bugs, include the model, base URL (redacted),
   and protocol. For features and new providers, open an issue before writing
   code so the design can be discussed.
2. **Create a branch** from `master`: `git checkout -b fix/anthropic-usage-merge`.
3. **Keep the layering honest.** `llm-trait` must stay free of heavy
   dependencies and must not depend on `llm-unified`. Protocol quirks belong in
   `src/protocol/*`; per-model facts belong in `src/model_registry/*`.
4. **Add a regression test** for every bug fix. Name tests after the behaviour
   they lock in (`max_tokens_with_profile_uses_profile_value`, not `test1`).
5. **Check the gates** before pushing:

   ```bash
   cargo fmt --all
   cargo clippy --all-targets -- -D warnings
   cargo test
   cargo doc --no-deps   # no broken intra-doc links
   ```

6. **Push and open a pull request** against `master`. Fill in the template;
   explain *why*, not just *what*.

## Commit Style

History uses [Conventional Commits](https://www.conventionalcommits.org/):

```text
fix(usage): add UsageInfo::merge and keep message_delta input_tokens
feat(registry): add profiles for llama.cpp endpoints
docs(readme): document protocol resolution order
refactor(protocol): extract SSE frame splitting into a helper
test(anthropic): cover truncated tool-call arguments
```

Scope to a component (`registry`, `protocol`, `usage`, `cli`, `docs`) where it
helps. One logical change per commit.

## Adding a Model or Provider

**Compatible with an existing protocol?** You usually need no code at all —
`base_url` plus a model name is enough, and unknown models fall back to the
protocol's safe default profile.

Add a registry module when the model has facts worth encoding (context window,
max output tokens, whether `reasoning_effort` is accepted, vision/tool support):

```rust
// src/model_registry/mybrand.rs
use super::ModelProfile;

pub fn profiles() -> Vec<(&'static str, ModelProfile)> {
    vec![("mybrand@openai", brand_openai_defaults())]
}

pub fn brand_prefixes() -> Vec<(&'static str, &'static str)> {
    vec![("my-", "mybrand")]
}
```

Register it in `ModelRegistry::builtin()` (`src/model_registry/registry.rs`) and
add a lookup test covering exact match, brand fallback, and prefix inference.

**New wire protocol?** Implement `RawAdapter` and add it under `src/protocol/`.
You own request building, SSE parsing, and tool-call assembly. Tests should
cover: happy path streaming, tool-call deltas split across chunks, HTTP error
bodies, and malformed/unknown frames (the fuzz targets are a good source of
inputs here).

## Fuzzing

The `fuzz` crate holds the harnesses; it is a workspace member but requires
nightly:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run parse_openai_response_fuzz -- -max_total_time=60
```

Pass a seed directory to start from real inputs rather than noise:

```bash
cargo +nightly fuzz run domain_matches_fuzz seeds/domain_matches_fuzz -- -max_total_time=60
```

### Fuzzing in CI

`.github/workflows/fuzz.yml` runs the real fuzzers, separate from `ci.yml`
because it needs nightly and an ASan build. Cost is one knob — `total_seconds`,
split evenly across the three targets:

| Trigger | Total fuzzing | Job timeout | Fails the build? |
|---------|---------------|-------------|------------------|
| Pull request | 60s | 30 min | no (advisory) |
| Weekly (Sun 03:00 UTC) | 900s | 60 min | yes |
| Manual dispatch | you choose | 60 min | yes |

A measured 30s run took 5m10s wall-clock: 38s compiling cargo-fuzz (then
cached), 3m37s for the ASan build, ~33s fuzzing. The floor is the build, not the
fuzzing — 10s per target still gave 5.5M executions total. PR runs report
failures as advisory so a nightly hiccup can't block a merge; the weekly
schedule is the hard gate. To fuzz longer: **Actions → fuzz → Run workflow**,
set `total_seconds`.

Seed corpora in `fuzz/seeds/<target>/` are committed. Generated input stays
ignored (`fuzz/corpus/`, `fuzz/artifacts/`).

If a fuzz run finds a crash: attach the reproducer from `fuzz/artifacts/`, add a
`tests/` regression case with the reduced input, and promote the input to a seed
file under `fuzz/seeds/<target>/` so CI keeps exploring from it.

## Publishing (maintainers)

```bash
cargo publish -p llm-trait     # interface crate first
cargo publish -p llm-unified
```

Both manifests need `license`, `repository`, `description`, and `keywords`, and
`llm-unified`'s dependency on `llm-trait` must carry a `version` requirement.
Tag releases as `v0.1.0` and update `CHANGELOG.md` in the same commit.

## Reporting Bugs

Use the issue template. Most valuable: exact model + base URL + protocol,
whether streaming or `chat()`, the raw response body (`--tools`/`raw` field or
`RUST_LOG=debug`), and expected vs actual. Please redact API keys.

## Security Issues

Do **not** open a public issue for vulnerabilities. See
[SECURITY.md](SECURITY.md).

## Style Notes

- `rustfmt` is authoritative — no manual alignment debates.
- Prefer explicit error context in `LlmError::config(...)` messages; they reach
  end users via the CLI.
- Public items need doc comments; `cargo doc` should be warning-free.
- Comments and documentation are written in English so the project stays
  accessible to contributors.
