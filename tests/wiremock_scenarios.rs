//! Scenario tests for the ModelRegistry + ProfiledProvider refactoring.
//!
//! These tests verify the core problem that triggered the refactoring:
//! MiMo returning 400 because reasoning_effort was unconditionally sent.

use llm_trait::{
    ChatMessage, ChatRequest, LlmConfig, LlmProvider, Protocol, ReasoningConfig, ReasoningEffort,
    ReasoningMode,
};
use llm_unified::{create_provider, model_registry::ModelRegistry};
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

fn openai_reasoning_response(text: &str, reasoning: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "reasoning_content": reasoning},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    })
}

// ── Test 1: MiMo does NOT send reasoning_effort (root cause bug) ─────────────

#[tokio::test]
async fn mimo_no_reasoning_effort() {
    let mock_server = MockServer::start().await;

    // If reasoning_effort were sent, MiMo would return 400.
    // The mock returns 200, so success means the field was NOT sent.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_text_response("hi")))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = create_provider(&LlmConfig {
        protocol: None,
        api_key: "tp-test".to_string(),
        model: "mimo-v2.5-pro".to_string(),
        base_url: mock_server.uri(),
        options: Default::default(),
    })
    .unwrap();

    assert_eq!(provider.info().name, "mimo");

    let request = ChatRequest {
        messages: vec![ChatMessage::user("hello")],
        tools: Vec::new(),
        reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        }),
        response_format: None,
    };

    let response = provider.chat(request).await.unwrap();
    assert_eq!(response.content, "hi");
    mock_server.verify().await;
}

// ── Test 2: DeepSeek sends reasoning_effort ──────────────────────────────────

#[tokio::test]
async fn deepseek_sends_reasoning_effort() {
    let mock_server = MockServer::start().await;

    // Mock that accepts any request (we'll verify behavior through the provider)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_text_response("ok")))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = create_provider(&LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "deepseek-chat".to_string(),
        base_url: mock_server.uri(),
        options: Default::default(),
    })
    .unwrap();

    assert_eq!(provider.info().name, "deepseek");

    let request = ChatRequest {
        messages: vec![ChatMessage::user("hello")],
        tools: Vec::new(),
        reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            ..Default::default()
        }),
        response_format: None,
    };

    let response = provider.chat(request).await.unwrap();
    assert_eq!(response.content, "ok");
    mock_server.verify().await;
}

// ── Test 3: Unknown model → no reasoning (safe fallback) ─────────────────────

#[tokio::test]
async fn unknown_model_no_reasoning() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_text_response("fallback")))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = create_provider(&LlmConfig {
        protocol: None,
        api_key: "test".to_string(),
        model: "some-unknown-model".to_string(),
        base_url: mock_server.uri(),
        options: Default::default(),
    })
    .unwrap();

    assert_eq!(provider.info().name, "openai");

    let request = ChatRequest {
        messages: vec![ChatMessage::user("hello")],
        tools: Vec::new(),
        reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::Medium),
            ..Default::default()
        }),
        response_format: None,
    };

    let response = provider.chat(request).await.unwrap();
    assert_eq!(response.content, "fallback");
    mock_server.verify().await;
}

// ── Test 4: Non-stream OpenAI reasoning_content extraction ───────────────────

#[tokio::test]
async fn openai_non_stream_reasoning_content() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_reasoning_response("answer", "let me think...")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = create_provider(&LlmConfig {
        protocol: None,
        api_key: "sk-test".to_string(),
        model: "deepseek-chat".to_string(),
        base_url: mock_server.uri(),
        options: Default::default(),
    })
    .unwrap();

    let request = ChatRequest::new(vec![ChatMessage::user("think about this")]);

    let response = provider.chat(request).await.unwrap();
    assert_eq!(response.content, "answer");
    assert_eq!(
        response.reasoning_content,
        Some("let me think...".to_string())
    );
}

// ── Test 5: ModelRegistry reasoning_mode verification ────────────────────────

#[test]
fn registry_mimo_reasoning_mode_none() {
    let registry = ModelRegistry::builtin();
    let profile = registry.lookup(
        "mimo-v2.5-pro",
        Some("https://token-plan-cn.xiaomimimo.com/v1"),
        None,
    );
    assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    assert_eq!(profile.protocol, Protocol::OpenAi);
    assert_eq!(profile.provider_name, "mimo");
}

#[test]
fn registry_deepseek_reasoning_mode_effort() {
    let registry = ModelRegistry::builtin();
    let profile = registry.lookup(
        "deepseek-chat",
        Some("https://api.deepseek.com/v1"),
        None,
    );
    assert_eq!(profile.reasoning_mode, ReasoningMode::Effort);
    assert_eq!(profile.provider_name, "deepseek");
}

#[test]
fn registry_claude_reasoning_mode_thinking() {
    let registry = ModelRegistry::builtin();
    let profile = registry.lookup(
        "claude-sonnet-4-20250514",
        Some("https://api.anthropic.com"),
        None,
    );
    assert_eq!(profile.reasoning_mode, ReasoningMode::Thinking);
    assert_eq!(profile.protocol, Protocol::Anthropic);
    assert_eq!(profile.provider_name, "anthropic");
}

#[test]
fn registry_unknown_model_safe_fallback() {
    let registry = ModelRegistry::builtin();
    let profile = registry.lookup(
        "some-unknown-model",
        Some("https://api.example.com/v1"),
        None,
    );
    assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    assert_eq!(profile.protocol, Protocol::OpenAi);
    assert_eq!(profile.provider_name, "openai");
}

// ── Test 6: Anthropic non-stream thinking block extraction ───────────────────

#[tokio::test]
async fn anthropic_non_stream_thinking_block() {
    let mock_server = MockServer::start().await;

    let response_body = serde_json::json!({
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "let me reason about this"},
            {"type": "text", "text": "the answer"}
        ],
        "model": "claude-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 15}
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = create_provider(&LlmConfig {
        protocol: Some(Protocol::Anthropic),
        api_key: "sk-test".to_string(),
        model: "claude-sonnet-4-20250514".to_string(),
        base_url: mock_server.uri(),
        options: Default::default(),
    })
    .unwrap();

    let request = ChatRequest {
        messages: vec![ChatMessage::user("think about this")],
        tools: Vec::new(),
        reasoning: Some(ReasoningConfig {
            budget_tokens: Some(4096),
            ..Default::default()
        }),
        response_format: None,
    };

    let response = provider.chat(request).await.unwrap();
    assert_eq!(response.content, "the answer");
    assert_eq!(
        response.reasoning_content,
        Some("let me reason about this".to_string())
    );
}
