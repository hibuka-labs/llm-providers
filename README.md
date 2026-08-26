# llm-providers

Unified LLM provider layer for the agent-base ecosystem.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│  Consumer (CLI, Web, agent-base)                                  │
│       │                                                           │
│       ▼                                                           │
│  ┌──────────────────────┐    ┌──────────────────────────────┐    │
│  │   llm-trait           │    │   llm-unified                │    │
│  │   (interface layer)   │◄───│   (implementation layer)     │    │
│  │                       │    │                              │    │
│  │  LlmProvider trait    │    │  AnthropicProtocol           │    │
│  │  RawAdapter trait     │    │  OpenAiProtocol              │    │
│  │  ChatRequest/Response │    │  GenericProvider<A>          │    │
│  │  LlmConfig/LlmBackend │    │  MimoProvider                │    │
│  │  Capabilities         │    │  DeepSeekProvider            │    │
│  │  UsageInfo            │    │  QwenProvider                │    │
│  └──────────────────────┘    │  factory (create_provider)    │    │
│                               └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Version | Description |
|-------|---------|-------------|
| `llm-trait` | 0.1.0 | Trait definitions and core types (zero heavy deps) |
| `llm-unified` | 0.1.0 | Concrete provider implementations |

## Quick Start

```rust
use llm_trait::{ChatMessage, ChatRequest, LlmConfig};
use llm_unified::create_provider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build config
    let config = LlmConfig {
        backend: "mimo".to_string(),
        protocol: None,
        api_key: "tp-xxx".to_string(),
        model: "mimo-v2.5-pro".to_string(),
        base_url: None,
        options: Default::default(),
    };

    // 2. Create provider
    let provider = create_provider(&config)?;

    // 3. Send a request
    let request = ChatRequest::new(vec![
        ChatMessage::user("Hello!"),
    ]);

    // Non-streaming
    let response = provider.chat(request.clone()).await?;
    println!("{}", response.content);

    // Streaming
    let mut stream = provider.stream(request).await?;
    while let Some(chunk) = stream.next().await {
        match chunk? {
            llm_trait::StreamChunk::Text(t) => print!("{}", t),
            llm_trait::StreamChunk::Stop { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
```

## Supported Providers

| Backend | Protocol | Notes |
|---------|----------|-------|
| `mimo` | Anthropic | Mimo AI (default) |
| `deepseek` | OpenAI | DeepSeek Chat / Reasoner |
| `qwen` | OpenAI | Alibaba Qwen via DashScope |
| `anthropic` | Anthropic | Claude API |
| custom + `protocol=openai` | OpenAI | Any OpenAI-compatible endpoint |
| custom + `protocol=anthropic` | Anthropic | Any Anthropic-compatible endpoint |

## Environment Variables

```bash
LLM_BACKEND=mimo           # Provider name
LLM_MODEL=mimo-v2.5-pro   # Model identifier
LLM_API_KEY=tp-xxx         # API key
LLM_BASE_URL=              # Custom base URL (optional)
LLM_PROTOCOL=              # Protocol override: openai | anthropic (optional)
```

```rust
use llm_unified::from_env;

let provider = from_env()?;
```

## CLI Tool

```bash
cargo run --bin llm-cli -- \
    --backend mimo \
    --model mimo-v2.5-pro \
    --api-key tp-xxx \
    --message "Hello" \
    --stream
```

## Adding a New Provider

See [docs/adapter-design.md](docs/adapter-design.md) §16.2 for the full checklist.

Short version:

1. **OpenAI-compatible?** → wrap `GenericProvider<OpenAiProtocol>`
2. **Anthropic-compatible?** → wrap `GenericProvider<AnthropicProtocol>`
3. **New protocol?** → implement `RawAdapter` trait
4. Register in `factory.rs`

## License

Internal use — buka-works project.
