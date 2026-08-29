//! Compile-check for the API surface used in README.md and docs/.
//!
//! Documentation examples are not compiled by `cargo test`, so a renamed method or
//! a changed signature silently turns the README into a bug report. This file
//! mirrors every snippet from the docs: if it stops matching reality, CI fails here.
//!
//! Keep it in sync when editing examples in README.md / docs/*.md.
#![allow(dead_code, unused_variables)]

use llm_trait::{
    Capabilities, ChatMessage, ChatRequest, LlmConfig, LlmError, LlmProvider, Protocol, RawAdapter,
    StreamChunk, UsageInfo,
};
use llm_unified::model_registry::{ModelProfile, ModelRegistry};
use llm_unified::{
    AnthropicProtocol, GenericProvider, OpenAiProtocol, create, create_provider, from_env,
};
use std::sync::Arc;

/// README → Quick Start
async fn readme_quickstart() -> Result<(), Box<dyn std::error::Error>> {
    let config = LlmConfig {
        protocol: None,
        api_key: std::env::var("LLM_API_KEY")?,
        model: "gpt-4o-mini".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        options: Default::default(),
    };

    let provider = create_provider(&config)?;

    let request = ChatRequest::new(vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Reply with one word."),
    ]);

    let response = provider.chat(request.clone()).await?;
    println!("{}", response.content);
    println!(
        "finish: {:?}, usage: {:?}",
        response.finish_reason, response.usage
    );

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

/// README → stream collection helpers (both consume the stream)
async fn readme_collect(
    provider: Arc<dyn LlmProvider>,
    request: ChatRequest,
) -> Result<(), LlmError> {
    let text = provider
        .stream(request.clone())
        .await?
        .collect_text()
        .await?;
    let full = provider.stream(request).await?.collect_response().await?;
    Ok(())
}

/// README → environment-based construction
fn readme_factories() -> Result<(), LlmError> {
    let from_env_provider = from_env()?;
    let created = create("sk-test", "gpt-4o", "https://api.openai.com/v1")?;
    Ok(())
}

/// docs/adding-a-provider.md → level 1 (no new code)
fn docs_level1(key: &str) -> Result<(), LlmError> {
    let config = LlmConfig {
        protocol: Some(Protocol::OpenAi),
        api_key: key.into(),
        model: "qwen2.5-coder:32b".into(),
        base_url: "http://localhost:11434/v1".into(),
        options: Default::default(),
    };
    let provider = create_provider(&config)?;
    Ok(())
}

/// docs/adding-a-provider.md → level 2 (registry profile fields)
fn docs_level2_fields() {
    let profile = ModelProfile {
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
        reasoning_mode: llm_trait::ReasoningMode::None,
        supported_extra_params: &[],
    };
    let registry = ModelRegistry::builtin();
    let resolved: ModelProfile =
        registry.lookup("my-Flagship-1", Some("https://api.mybrand.com/v1"), None);
}

/// docs/adding-a-provider.md → level 3 (wrap an adapter, inject transport)
fn docs_level3(
    adapter: Box<dyn RawAdapter>,
    client: Arc<dyn llm_trait::HttpClient>,
    config: &LlmConfig,
) {
    let wrapped = GenericProvider::new(Box::new(OpenAiProtocol::from_config(config)));
    let injected = GenericProvider::with_http_client(adapter, client, Default::default());
    let _: &dyn RawAdapter = injected.adapter();
}

/// README / docs → error inspection and finish-reason parsing
fn docs_errors() {
    let err = LlmError::api(429, "slow down");
    assert_eq!(err.status(), Some(429));
    assert!(LlmError::config("missing key").status().is_none());

    let by_inherent = llm_trait::FinishReason::from_str("max_tokens");
    let by_from_str: llm_trait::FinishReason = "length".parse().unwrap();
    assert_eq!(by_inherent, by_from_str);

    let mut usage = UsageInfo::default();
    usage.merge(&UsageInfo {
        prompt_tokens: Some(10),
        ..Default::default()
    });
    let _ = AnthropicProtocol::new("k", "m", Some("http://localhost"));
}

#[test]
fn doc_examples_stay_compiled() {
    // The value of this file is that it compiles; assert nothing at runtime.
}
