//! Wiremock integration tests for DeepSeek and Qwen via create_provider().
//!
//! Covers:
//! - DeepSeek streaming + non-streaming calls
//! - Qwen streaming + non-streaming calls
//! - Provider info() returns correct name (from registry)
//! - create_provider() routes correctly

use llm_trait::{ChatMessage, ChatRequest, LlmProvider, StreamChunk};
use llm_unified::{create_provider, GenericProvider, OpenAiProtocol};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn openai_text_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn sse_data(data: &str) -> String {
    format!("data: {data}\n\n")
}

fn sse_text_body(text: &str) -> String {
    let mut body = String::new();
    body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
    }).to_string()));
    body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }).to_string()));
    body.push_str("data: [DONE]\n\n");
    body
}

// ── DeepSeek tests ───────────────────────────────────────────────────────────

#[test]
fn deepseek_provider_info() {
    let config = llm_trait::LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "deepseek-chat".to_string(),
        base_url: "https://api.deepseek.com/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    let info = provider.info();
    assert_eq!(info.name, "deepseek");
    assert_eq!(info.model, "deepseek-chat");
}

#[tokio::test]
async fn deepseek_chat_returns_response() {
    let server = MockServer::start().await;

    let body = openai_text_response("Hello from DeepSeek!");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "deepseek-chat", Some(&server.uri()));
    let provider = GenericProvider::new(Box::new(protocol));
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Hello from DeepSeek!");
    assert_eq!(response.usage.prompt_tokens, Some(10));
}

#[tokio::test]
async fn deepseek_stream_collects_text() {
    let server = MockServer::start().await;

    let sse_body = sse_text_body("DeepSeek streams!");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "deepseek-chat", Some(&server.uri()));
    let provider = GenericProvider::new(Box::new(protocol));
    let request = ChatRequest::new(vec![ChatMessage::user("test")]);
    let stream = provider.stream(request).await.unwrap();
    let text = stream.collect_text().await.unwrap();

    assert_eq!(text, "DeepSeek streams!");
}

// ── Qwen tests ───────────────────────────────────────────────────────────────

#[test]
fn qwen_provider_info() {
    let config = llm_trait::LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "qwen-plus".to_string(),
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    let info = provider.info();
    assert_eq!(info.name, "qwen");
    assert_eq!(info.model, "qwen-plus");
}

#[tokio::test]
async fn qwen_chat_returns_response() {
    let server = MockServer::start().await;

    let body = openai_text_response("Hello from Qwen!");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "qwen-plus", Some(&server.uri()));
    let provider = GenericProvider::new(Box::new(protocol));
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Hello from Qwen!");
    assert_eq!(response.usage.prompt_tokens, Some(10));
}

#[tokio::test]
async fn qwen_stream_collects_text() {
    let server = MockServer::start().await;

    let sse_body = sse_text_body("Qwen streams!");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "qwen-plus", Some(&server.uri()));
    let provider = GenericProvider::new(Box::new(protocol));
    let request = ChatRequest::new(vec![ChatMessage::user("test")]);
    let stream = provider.stream(request).await.unwrap();
    let text = stream.collect_text().await.unwrap();

    assert_eq!(text, "Qwen streams!");
}

// ── Factory routing tests ────────────────────────────────────────────────────

#[test]
fn factory_routes_deepseek() {
    let config = llm_trait::LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "deepseek-chat".to_string(),
        base_url: "https://api.deepseek.com/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    // After refactor: registry returns "deepseek" as provider name
    assert_eq!(provider.info().name, "deepseek");
}

#[test]
fn factory_routes_qwen() {
    let config = llm_trait::LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "qwen-plus".to_string(),
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    // After refactor: registry returns "qwen" as provider name
    assert_eq!(provider.info().name, "qwen");
}

#[test]
fn factory_routes_openai_protocol() {
    let config = llm_trait::LlmConfig {
        protocol: Some(llm_trait::Protocol::OpenAi),
        api_key: "sk-test".to_string(),
        model: "gpt-4o".to_string(),
        base_url: "https://custom.api.com/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    assert_eq!(provider.info().name, "openai");
}

// ── MiMo routing tests ───────────────────────────────────────────────────

#[test]
fn factory_routes_mimo() {
    let config = llm_trait::LlmConfig {
        protocol: None,
        api_key: "tp-test".to_string(),
        model: "mimo-v2.5-pro".to_string(),
        base_url: "https://token-plan-cn.xiaomimimo.com/v1".to_string(),
        options: Default::default(),
    };
    let provider = create_provider(&config).unwrap();
    assert_eq!(provider.info().name, "mimo");
}
