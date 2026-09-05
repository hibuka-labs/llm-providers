# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |
| < 0.1   | ❌ |

This project is pre-1.0. Security fixes land on the latest release and, when
practical, on `main`.

## Why This Matters Here

`llm-providers` handles API credentials and parses untrusted network responses.
The security-relevant surface is small but real:

- **Credential handling** — API keys are placed in HTTP headers by
  `ReqwestHttpClient`. Anything that logs a request, an error body, or a panic
  message could leak a key. `Debug` for `LlmConfig` prints the key as `***`, and
  `Debug` for `RawRequest` redacts values of authentication-related header names
  (`authorization`, `x-api-key`, anything containing `token`). Keep it that way:
  never add a derive that would resurrect the plaintext key.
- **Untrusted input** — `RawAdapter::parse_response` and the SSE parsers consume
  attacker-influenced JSON. Malformed or adversarial payloads must produce an
  `LlmError`, never a panic or unbounded memory growth. The `fuzz/` targets exist
  specifically for this.
- **Tool-call assembly** — streamed tool-call fragments are concatenated before
  being handed to callers as JSON. Truncated or nested payloads must not be
  silently repaired into valid-looking arguments.
- **Endpoint trust** — `base_url` is caller-supplied. The URL/domain inference in
  `ModelRegistry` decides which protocol adapter handles a request, so hostname
  matching must respect domain boundaries (`anthropic.com.evil.com` must not be
  treated as Anthropic).

## Reporting a Vulnerability

Please report suspected vulnerabilities **privately**, not via a public issue:

- GitHub private vulnerability report: https://github.com/hibuka-labs/llm-providers/security/advisories/new
- Or email: chenkangzeng@163.com

Include: affected version(s), a description, and if possible a minimal repro or
a failing test. Public PoCs help nobody until a fix exists.

## Timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | 3 business days |
| Preliminary assessment | 7 business days |
| Fix + release, or explanation of delay | 30 days, depending on severity |

Coordinated disclosure is preferred: we will agree on a publication window
before a fix ships, and credit reporters who want it.

## Hardening Checklist for Contributors

- Never log `api_key`, headers, or full request bodies.
- Propagate malformed-input errors as `LlmError`; avoid `unwrap()`/`expect()` on
  values derived from network data.
- Bound anything that grows with input size (buffers, retry counts, chunk counts).
- Add a fuzz regression case when a parser edge is fixed.
