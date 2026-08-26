//! Wiremock integration tests for Anthropic protocol.
//!
//! Covers the Phase 2 acceptance criteria from docs/adapter-design.md §15.3:
//! - Non-streaming `chat()` correct parsing
//! - Streaming `stream()` correct SSE event parsing
//! - Error handling (401/429/500) returns correct LlmError
//! - Retry logic (429 auto-retry, 401 no retry)

use llm_trait::{ChatMessage, ChatRequest, LlmError, LlmProvider, StreamChunk};
use llm_unified::{AnthropicProtocol, GenericProvider};
use futures_util::StreamExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a `GenericProvider<AnthropicProtocol>` pointing at the mock server.
fn make_provider(base_url: &str) -> GenericProvider<AnthropicProtocol> {
    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(base_url));
    GenericProvider::new(protocol)
}

/// Anthropic non-streaming success response body.
fn anthropic_text_response(text: &str, input_tokens: u32, output_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "test-model",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

/// Anthropic error response body.
fn anthropic_error_response(err_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": err_type,
            "message": message
        }
    })
}

/// Build an SSE event string in the format expected by `eventsource-stream`.
fn sse_event(event_type: &str, data: &str) -> String {
    format!("event: {event_type}\ndata: {data}\n\n")
}

// ── 1. Non-streaming chat() ──────────────────────────────────────────────────

#[tokio::test]
async fn chat_returns_parsed_response() {
    let server = MockServer::start().await;

    let body = anthropic_text_response("Bonjour!", 12, 8);
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "sk-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("Say bonjour")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Bonjour!");
    assert!(response.tool_calls.is_empty());
    assert_eq!(response.usage.prompt_tokens, Some(12));
    assert_eq!(response.usage.completion_tokens, Some(8));
}

#[tokio::test]
async fn chat_parses_tool_use_response() {
    let server = MockServer::start().await;

    let body = serde_json::json!({
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "rust"}}
        ],
        "model": "test-model",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    });
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("search for rust")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Let me search.");
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "search");
    assert_eq!(response.tool_calls[0].id, "call_1");
}

// ── 2. Streaming stream() ────────────────────────────────────────────────────

#[tokio::test]
async fn stream_collects_text_from_sse_events() {
    let server = MockServer::start().await;

    // Build SSE body: message_start → content_block_delta × 2 → message_delta → message_stop
    // Each data payload must include "type" field for serde(tag = "type") deserialization.
    let mut sse_body = String::new();
    sse_body.push_str(&sse_event(
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "usage": {"input_tokens": 15, "output_tokens": 0}
            }
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world!"}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 3}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event("message_stop", r#"{"type": "message_stop"}"#));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("greet")]);
    let stream = provider.stream(request).await.unwrap();
    let text = stream.collect_text().await.unwrap();

    assert_eq!(text, "Hello world!");
}

#[tokio::test]
async fn stream_extracts_usage_info() {
    let server = MockServer::start().await;

    // Note: collect_response() does `usage = u` (full replacement) on each Usage chunk.
    // The last Usage chunk (from MessageDelta) overwrites the one from MessageStart.
    // So prompt_tokens from MessageStart is lost in the final ChatResponse.
    // We test chunk-level extraction separately in stream_emits_usage_chunks.
    let mut sse_body = String::new();
    sse_body.push_str(&sse_event(
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_u",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "usage": {"input_tokens": 20, "output_tokens": 0}
            }
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 7}
        })
        .to_string(),
    ));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("test")]);
    let stream = provider.stream(request).await.unwrap();
    let response = stream.collect_response().await.unwrap();

    assert_eq!(response.content, "ok");
    // Last Usage chunk (MessageDelta) overwrites; prompt_tokens is None
    assert_eq!(response.usage.prompt_tokens, None);
    assert_eq!(response.usage.completion_tokens, Some(7));
}

#[tokio::test]
async fn stream_emits_usage_chunks_with_prompt_tokens() {
    // Verify that the first Usage chunk (from message_start) carries prompt_tokens.
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    sse_body.push_str(&sse_event(
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_uc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "usage": {"input_tokens": 42, "output_tokens": 0}
            }
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "x"}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 5}
        })
        .to_string(),
    ));

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("test")]);
    let mut stream = provider.stream(request).await.unwrap();

    // Collect individual chunks and inspect Usage chunks
    let mut usage_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Usage(u)) = chunk {
            usage_chunks.push(u);
        }
    }

    assert_eq!(usage_chunks.len(), 2, "Expected 2 Usage chunks (message_start + message_delta)");
    assert_eq!(usage_chunks[0].prompt_tokens, Some(42), "First chunk has prompt_tokens from message_start");
    assert_eq!(usage_chunks[0].completion_tokens, Some(0));
    assert_eq!(usage_chunks[1].prompt_tokens, None, "Second chunk (message_delta) has no prompt_tokens");
    assert_eq!(usage_chunks[1].completion_tokens, Some(5));
}

// ── 3. Error handling ────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_401_returns_api_error() {
    let server = MockServer::start().await;

    let body = anthropic_error_response("authentication_error", "Invalid API key");
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let err = provider.chat(request).await.unwrap_err();

    match &err {
        LlmError::LlmApi { status, .. } => assert_eq!(*status, 401),
        other => panic!("Expected LlmApi, got: {other:?}"),
    }
}

#[tokio::test]
async fn chat_429_returns_api_error() {
    let server = MockServer::start().await;

    // Always return 429 so retries exhaust and we get the error.
    let body = anthropic_error_response("rate_limit_error", "Rate limit exceeded");
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&body))
        .expect(4) // 1 initial + 3 retries
        .mount(&server)
        .await;

    // Use a provider with short retry delay to keep the test fast.
    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(protocol, config);

    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let err = provider.chat(request).await.unwrap_err();

    match &err {
        LlmError::LlmApi { status, .. } => assert_eq!(*status, 429),
        other => panic!("Expected LlmApi, got: {other:?}"),
    }
}

#[tokio::test]
async fn chat_500_returns_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(500).set_body_string("Internal Server Error"),
        )
        .expect(4) // 1 initial + 3 retries (5xx is retryable)
        .mount(&server)
        .await;

    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(protocol, config);

    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let err = provider.chat(request).await.unwrap_err();

    match &err {
        LlmError::LlmApi { status, .. } => assert_eq!(*status, 500),
        other => panic!("Expected LlmApi, got: {other:?}"),
    }
}

// ── 4. Retry logic ───────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_429_retries_then_succeeds() {
    let server = MockServer::start().await;

    // First 2 requests → 429, third → 200.
    let error_body = anthropic_error_response("rate_limit_error", "Too many requests");
    let success_body = anthropic_text_response("Recovered!", 5, 3);

    // We need ordered expectations. wiremock matches in registration order,
    // so mount the success first (lowest priority) then the 429s.
    // Actually wiremock 0.6 matches the *first* matching mock, so we register
    // 429 stubs with explicit expectations and a fallback 200.
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_body))
        .up_to_n_times(2)
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(protocol, config);

    let request = ChatRequest::new(vec![ChatMessage::user("retry test")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Recovered!");
}

#[tokio::test]
async fn chat_401_does_not_retry() {
    let server = MockServer::start().await;

    let body = anthropic_error_response("authentication_error", "Bad key");
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&body))
        .expect(1) // Exactly 1 — no retries
        .mount(&server)
        .await;

    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(protocol, config);

    let request = ChatRequest::new(vec![ChatMessage::user("auth test")]);
    let err = provider.chat(request).await.unwrap_err();

    match &err {
        LlmError::LlmApi { status, .. } => assert_eq!(*status, 401),
        other => panic!("Expected LlmApi, got: {other:?}"),
    }
    // wiremock will verify exactly 1 request was made (via .expect(1))
}
