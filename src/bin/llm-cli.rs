use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use llm_trait::{ChatMessage, ChatRequest, LlmConfig, StreamChunk};

#[derive(Parser)]
#[command(name = "llm-cli", about = "LLM unified CLI verification tool")]
struct Cli {
    /// Model name
    #[arg(long, env = "LLM_MODEL")]
    model: String,

    /// API key
    #[arg(long, env = "LLM_API_KEY")]
    api_key: String,

    /// Base URL (required)
    #[arg(long, env = "LLM_BASE_URL")]
    base_url: String,

    /// Protocol override (anthropic, openai)
    #[arg(long, env = "LLM_PROTOCOL")]
    protocol: Option<String>,

    /// User message (required)
    #[arg(long)]
    message: String,

    /// System prompt
    #[arg(long)]
    system: Option<String>,

    /// Enable streaming output
    #[arg(long)]
    stream: bool,

    /// Tool definitions JSON file
    #[arg(long)]
    tools: Option<PathBuf>,

    /// Max output tokens
    #[arg(long, default_value = "4096")]
    max_tokens: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Build config
    let protocol = cli.protocol.as_deref().and_then(|s| s.parse().ok());
    let config = LlmConfig {
        protocol,
        api_key: cli.api_key.clone(),
        model: cli.model.clone(),
        base_url: cli.base_url.clone(),
        options: Default::default(),
    };

    // Create provider
    let provider = llm_unified::create_provider(&config)?;

    eprintln!("Provider: {}", provider.info().name);
    eprintln!("Model: {}", provider.info().model);
    eprintln!();

    // Build messages
    let mut messages = Vec::new();
    if let Some(ref sys) = cli.system {
        messages.push(ChatMessage::system(sys));
    }
    messages.push(ChatMessage::user(&cli.message));

    // Build tools
    let tools = if let Some(ref path) = cli.tools {
        let content = std::fs::read_to_string(path)?;
        let tools_value: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(arr) = tools_value.as_array() {
            arr.clone()
        } else {
            vec![tools_value]
        }
    } else {
        vec![]
    };

    // Build request
    let request = ChatRequest::new(messages).with_tools(tools);

    if cli.stream {
        // Streaming output
        eprintln!("[streaming mode]");
        eprintln!();
        let mut stream = provider.stream(request).await?;
        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Text(text) => {
                    print!("{}", text);
                    std::io::stdout().flush()?;
                }
                StreamChunk::Thought(thinking) => {
                    eprintln!("[thinking] {}", thinking);
                }
                StreamChunk::ThinkingSignature(sig) => {
                    eprintln!("[thinking signature] {}", &sig[..20.min(sig.len())]);
                }
                StreamChunk::ToolCall(call) => {
                    eprintln!("[tool call] {}", call);
                }
                StreamChunk::Usage(usage) => {
                    eprintln!("[usage] {:?}", usage);
                }
                StreamChunk::Error(msg) => {
                    eprintln!("[stream error] {}", msg);
                    break;
                }
                StreamChunk::Stop { finish_reason } => {
                    eprintln!();
                    eprintln!("[stop] reason: {:?}", finish_reason);
                    break;
                }
            }
        }
    } else {
        // Non-streaming output
        eprintln!("[non-streaming mode]");
        eprintln!();
        let response = provider.chat(request).await?;
        println!("{}", response.content);
        eprintln!();
        eprintln!(
            "[usage] prompt: {:?}, completion: {:?}, total: {:?}",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        );
        eprintln!("[finish_reason] {:?}", response.finish_reason);
        if !response.tool_calls.is_empty() {
            eprintln!("[tool_calls]");
            for tc in &response.tool_calls {
                eprintln!("  - {} ({}): {}", tc.name, tc.id, tc.arguments);
            }
        }
    }

    Ok(())
}
