// Minimal end-to-end example matching the README quick start.
//
// Run it with a real endpoint:
//   export LLM_API_KEY=sk-...
//   cargo run --example quickstart
//
// Or point it at a local proxy:
//   LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5 cargo run --example quickstart

use llm_trait::{ChatMessage, ChatRequest, LlmConfig, StreamChunk};
use llm_unified::create_provider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure: API key + model + base URL.
    //    `protocol: None` lets the model registry infer it from the URL/model.
    let config = LlmConfig {
        protocol: None,
        api_key: std::env::var("LLM_API_KEY").unwrap_or_default(),
        model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
        base_url: std::env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        options: Default::default(),
    };

    // 2. Build the provider.
    let provider = create_provider(&config)?;
    let info = provider.info();
    eprintln!("provider: {} (model {})", info.name, info.model);
    eprintln!("capabilities: {:?}", provider.capabilities());

    let request = ChatRequest::new(vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Reply with exactly one word."),
    ]);

    // 3a. Non-streaming.
    let response = provider.chat(request.clone()).await?;
    println!("[chat] {}", response.content);
    println!(
        "[chat] finish={:?} usage={:?}",
        response.finish_reason, response.usage
    );

    // 3b. Streaming.
    let mut stream = provider.stream(request).await?;
    print!("[stream] ");
    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk::Text(text) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            StreamChunk::Thought(thinking) => eprintln!("\n[thinking] {thinking}"),
            StreamChunk::ToolCall(call) => eprintln!("\n[tool call] {call}"),
            StreamChunk::Stop { finish_reason } => {
                eprintln!("\n[stream] stopped: {finish_reason:?}");
                break;
            }
            StreamChunk::Error(message) => {
                eprintln!("\n[stream] error: {message}");
                break;
            }
            StreamChunk::Usage(usage) => eprintln!("\n[usage] {usage:?}"),
            StreamChunk::ThinkingSignature(_) => {}
        }
    }

    Ok(())
}
