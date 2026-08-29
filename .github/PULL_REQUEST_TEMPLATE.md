<!--
Provider/protocol bugs: please include model, base URL (redacted), protocol, and
the raw provider response — a PR body without those is hard to review.
-->

## What does this PR do?

## Why?

## How was it tested?

- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo fmt --all --check`
- [ ] Added/updated tests for the change
- [ ] Updated `CHANGELOG.md` under *Unreleased* (if user-visible)
- [ ] Updated docs/README (if behaviour or API changed)

## Checklist

- [ ] `llm-trait` still has no dependency on `llm-unified` or any concrete provider
- [ ] No new panics reachable from untrusted input (network responses, JSON bodies)
- [ ] No API keys, internal hostnames, or customer data added
- [ ] Breaking changes are called out below, with a migration note

## Related issues

Closes #
