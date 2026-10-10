//! Wiremock integration tests for Anthropic protocol.
//!
//! Covers:
//! - Non-streaming `chat()` correct parsing
//! - Streaming `stream()` correct SSE event parsing
//! - Error handling (401/429/500) returns correct LlmError
//! - Retry logic (429 auto-retry, 401 no retry)

use llm_trait::{ChatMessage, ChatRequest, LlmError, LlmProvider, StreamChunk};
use llm_unified::{AnthropicProtocol, GenericProvider};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_provider(base_url: &str) -> GenericProvider {
    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(base_url));
    GenericProvider::new(Box::new(protocol))
}

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

fn anthropic_error_response(err_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": err_type,
            "message": message
        }
    })
}

fn sse_event(event_type: &str, data: &str) -> String {
    format!("event: {event_type}\ndata: {data}\n\n")
}

// ── 1. Non-streaming chat() ──────────────────────────────────────────────────

#[tokio::test]
async fn chat_returns_parsed_response() {
    let server = MockServer::start().await;

    let body = anthropic_text_response("Bonjour!", 12, 8);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
        .and(path("/v1/messages"))
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
        .and(path("/v1/messages"))
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
async fn stream_tolerates_trailing_openai_done_marker() {
    // 防御测试：上游（如 llm-proxy 中继）可能在合法的 message_stop 之后
    // 补发 OpenAI 风格的 `data: [DONE]` 终止符。[DONE] 不是合法 JSON，
    // 解析器应跳过它，而不是报 "Failed to parse SSE event" 整流失败。
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    sse_body.push_str(&sse_event(
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_done",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "usage": {"input_tokens": 5, "output_tokens": 0}
            }
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello world!"}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 2}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event("message_stop", r#"{"type": "message_stop"}"#));
    // 事故根源：代理补发的 OpenAI 风格终止符
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
    let mut stream = provider.stream(request).await.unwrap();

    // 手动排空整个流（collect_text 遇 Stop 即断，看不到 Stop 之后的错误 chunk）
    let mut text = String::new();
    let mut errors = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamChunk::Text(t)) => text.push_str(&t),
            Ok(_) => {}
            Err(e) => errors.push(e.to_string()),
        }
    }

    assert!(
        errors.is_empty(),
        "trailing [DONE] must be tolerated, got errors: {errors:?}"
    );
    assert_eq!(text, "Hello world!");
}

/// 构造一条完整的合法 Anthropic SSE 流（供 keep-alive / 坏 JSON 测试复用）。
fn anthropic_sse_happy_body() -> String {
    let mut sse_body = String::new();
    sse_body.push_str(&sse_event(
        "message_start",
        &serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_keepalive",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "usage": {"input_tokens": 5, "output_tokens": 0}
            }
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "content_block_delta",
        &serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello world!"}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event(
        "message_delta",
        &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 2}
        })
        .to_string(),
    ));
    sse_body.push_str(&sse_event("message_stop", r#"{"type": "message_stop"}"#));
    sse_body
}

/// 排空流并返回 (拼接文本, 错误列表)——collect_text 遇 Stop 即断，
/// 看不到 Stop 之后（或中途）的错误 chunk，需要手动排空。
async fn drain_stream(stream: &mut llm_trait::ChatStream) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut errors = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamChunk::Text(t)) => text.push_str(&t),
            Ok(_) => {}
            Err(e) => errors.push(e.to_string()),
        }
    }
    (text, errors)
}

#[tokio::test]
async fn stream_tolerates_processing_keepalive() {
    // 防御测试：MiMo 等上游会以 data 字段夹带注释式心跳 `data: : PROCESSING`。
    // 它不是合法 JSON，解析器应跳过它，而不是报 "Failed to parse SSE event"
    // 整流失败、白烧 turn_retry 整轮重连（实测 1000/2000/4000ms 三次全废）。
    let server = MockServer::start().await;

    let mut sse_body = anthropic_sse_happy_body();
    // 事故根源：上游在 message_start 之后、正式 delta 之前插入心跳
    sse_body.insert_str(
        sse_body.find("event: content_block_delta").unwrap(),
        "data: : PROCESSING\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
    let mut stream = provider.stream(request).await.unwrap();

    let (text, errors) = drain_stream(&mut stream).await;
    assert!(
        errors.is_empty(),
        "keep-alive `: PROCESSING` must be tolerated, got errors: {errors:?}"
    );
    assert_eq!(text, "Hello world!");
}

#[tokio::test]
async fn stream_malformed_json_still_fatal() {
    // 不变式：只有已知哨兵（[DONE] / `: ...`）可跳过；
    // 真正的坏 JSON 必须继续致命，否则会静默丢事件。
    let server = MockServer::start().await;

    let mut sse_body = anthropic_sse_happy_body();
    sse_body.insert_str(
        sse_body.find("event: content_block_delta").unwrap(),
        "data: {broken json\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
    let mut stream = provider.stream(request).await.unwrap();

    let (_text, errors) = drain_stream(&mut stream).await;
    assert!(
        errors
            .iter()
            .any(|e| e.contains("Failed to parse SSE event")),
        "genuinely malformed JSON must stay fatal, got errors: {errors:?}"
    );
}

#[tokio::test]
async fn stream_usage_accumulates_correctly() {
    // With the fix: prompt_tokens from message_start is preserved
    // even when message_delta only has completion_tokens
    let server = MockServer::start().await;

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
        .and(path("/v1/messages"))
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
    // After fix: prompt_tokens is preserved from message_start
    assert_eq!(response.usage.prompt_tokens, Some(20));
    assert_eq!(response.usage.completion_tokens, Some(7));
}

// ── 3. Error handling ────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_401_returns_api_error() {
    let server = MockServer::start().await;

    let body = anthropic_error_response("authentication_error", "Invalid API key");
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
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
async fn chat_429_retries_then_succeeds() {
    let server = MockServer::start().await;

    let error_body = anthropic_error_response("rate_limit_error", "Too many requests");
    let success_body = anthropic_text_response("Recovered!", 5, 3);

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_body))
        .up_to_n_times(2)
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        // 测试里压缩限流退避，验证的是「会重试」语义而非真实等待时长
        rate_limit_retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(Box::new(protocol), config);

    let request = ChatRequest::new(vec![ChatMessage::user("retry test")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "Recovered!");
}

#[tokio::test]
async fn chat_500_returns_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .expect(4)
        .mount(&server)
        .await;

    let protocol = AnthropicProtocol::new("sk-test", "test-model", Some(&server.uri()));
    let config = llm_unified::generic::ProviderConfig {
        max_retries: 3,
        retry_delay: std::time::Duration::from_millis(1),
        ..Default::default()
    };
    let provider = GenericProvider::with_config(Box::new(protocol), config);

    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let err = provider.chat(request).await.unwrap_err();

    match &err {
        LlmError::LlmApi { status, .. } => assert_eq!(*status, 500),
        other => panic!("Expected LlmApi, got: {other:?}"),
    }
}
