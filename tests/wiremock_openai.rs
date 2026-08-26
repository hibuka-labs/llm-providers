//! Wiremock integration tests for OpenAI Chat Completions protocol.
//!
//! Covers the Phase 4a acceptance criteria from docs/adapter-design.md §15.5.1:
//! - Non-streaming `chat()` correct parsing
//! - Streaming `stream()` correct SSE event parsing
//! - `data: [DONE]` correctly terminates stream
//! - Usage information correctly extracted
//! - Tool call incremental assembly
//! - Error handling (401/429/500)

use llm_trait::{ChatMessage, ChatRequest, LlmError, LlmProvider, StreamChunk};
use llm_unified::{GenericProvider, OpenAiProtocol};
use futures_util::StreamExt;
use wiremock::matchers::{header, method, path, body_string_contains};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a `GenericProvider<OpenAiProtocol>` pointing at the mock server.
fn make_provider(base_url: &str) -> GenericProvider<OpenAiProtocol> {
    let protocol = OpenAiProtocol::new("sk-test", "test-model", Some(base_url));
    GenericProvider::new(protocol)
}

/// OpenAI non-streaming success response body.
fn openai_text_response(text: &str, prompt_tokens: u32, completion_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

/// OpenAI error response body.
fn openai_error_response(err_type: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "type": err_type,
            "message": message
        }
    })
}

/// Build a single SSE data line (OpenAI format: `data: {...}\n\n`).
fn sse_data(data: &str) -> String {
    format!("data: {data}\n\n")
}

// ── 1. Non-streaming chat() ──────────────────────────────────────────────────

#[tokio::test]
async fn chat_returns_parsed_response() {
    let server = MockServer::start().await;

    let body = openai_text_response("Bonjour!", 12, 8);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
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
    assert_eq!(response.usage.total_tokens, Some(20));
}

#[tokio::test]
async fn chat_parses_tool_calls_response() {
    let server = MockServer::start().await;

    let body = serde_json::json!({
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\":\"rust\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("search for rust")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "");
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "search");
    assert_eq!(response.tool_calls[0].id, "call_1");
    assert_eq!(response.tool_calls[0].arguments, "{\"q\":\"rust\"}");
    assert_eq!(response.finish_reason, llm_trait::FinishReason::ToolCalls);
}

// ── 2. Streaming stream() ────────────────────────────────────────────────────

#[tokio::test]
async fn stream_collects_text_from_sse_events() {
    let server = MockServer::start().await;

    // OpenAI SSE: data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n
    let mut sse_body = String::new();
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": "Hello "},
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": "world!"},
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }).to_string()));
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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
async fn stream_handles_done_termination() {
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-done",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": "ok"},
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-done",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }).to_string()));
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

    let mut stop_count = 0;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::Stop { .. }) = chunk {
            stop_count += 1;
        }
    }
    // At least one Stop from the finish_reason, one from [DONE]
    assert!(stop_count >= 1, "Expected at least 1 Stop chunk, got {stop_count}");
}

#[tokio::test]
async fn stream_extracts_usage_from_final_chunk() {
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"content": "hi"},
            "finish_reason": null
        }]
    }).to_string()));
    // Final chunk with usage info
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-usage",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 7,
            "total_tokens": 22
        }
    }).to_string()));
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

    assert_eq!(response.content, "hi");
    assert_eq!(response.usage.prompt_tokens, Some(15));
    assert_eq!(response.usage.completion_tokens, Some(7));
}

#[tokio::test]
async fn stream_increments_tool_call_assembly() {
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    // First chunk: tool call start with id and name
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "search", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Incremental argument fragments
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"q\":"}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "\"test\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }).to_string()));
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("search")]);
    let mut stream = provider.stream(request).await.unwrap();

    let mut tool_call_chunks = 0;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::ToolCall(_)) = chunk {
            tool_call_chunks += 1;
        }
    }
    assert_eq!(tool_call_chunks, 3, "Expected 3 tool call chunks (start + 2 arg fragments)");
}

// ── 3. Error handling ────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_401_returns_api_error() {
    let server = MockServer::start().await;

    let body = openai_error_response("invalid_request_error", "Invalid API key");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
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

    let error_body = openai_error_response("rate_limit_error", "Too many requests");
    let success_body = openai_text_response("Recovered!", 5, 3);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_body))
        .up_to_n_times(2)
        .expect(2)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&success_body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "test-model", Some(&server.uri()));
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

    let body = openai_error_response("invalid_request_error", "Bad key");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "test-model", Some(&server.uri()));
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
}

#[tokio::test]
async fn chat_500_returns_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .expect(4) // 1 initial + 3 retries
        .mount(&server)
        .await;

    let protocol = OpenAiProtocol::new("sk-test", "test-model", Some(&server.uri()));
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

// ── 4. Tool definitions in request ─────────────────────────────────────────

#[tokio::test]
async fn chat_sends_tools_in_request() {
    let server = MockServer::start().await;

    let body = openai_text_response("No tools needed", 10, 5);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(r#""name":"get_weather""#))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    })];
    let request = ChatRequest::new(vec![ChatMessage::user("What's the weather?")])
        .with_tools(tools);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.content, "No tools needed");
}

#[tokio::test]
async fn chat_sends_multiple_tools_in_request() {
    let server = MockServer::start().await;

    let body = openai_text_response("ok", 10, 5);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_string_contains(r#""name":"get_weather""#))
        .and(body_string_contains(r#""name":"search""#))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let tools = vec![
        serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}}
        }),
        serde_json::json!({
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}}
        }),
    ];
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(tools);
    provider.chat(request).await.unwrap();
}

// ── 5. Multi-turn tool-call loop ──────────────────────────────────────────

#[tokio::test]
async fn chat_tool_call_multi_turn() {
    // Simulates a 2-call loop:
    //   Round 1: user asks → model returns tool_calls
    //   Round 2: tool result sent → model returns final text
    let server = MockServer::start().await;

    // Round 1 response: tool call
    let tool_call_body = serde_json::json!({
        "id": "chatcmpl-tc1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": r#"{"city":"Shanghai"}"#}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
    });

    // Round 2 response: final text
    let final_body = openai_text_response("Shanghai is sunny!", 30, 10);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&tool_call_body))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&final_body))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());

    // Round 1
    let request = ChatRequest::new(vec![ChatMessage::user("What's the weather in Shanghai?")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "get_weather");
    assert_eq!(response.finish_reason, llm_trait::FinishReason::ToolCalls);

    // Round 2: send tool result back
    let request2 = ChatRequest::new(vec![
        ChatMessage::user("What's the weather in Shanghai?"),
        ChatMessage::assistant_tool_call(&response.tool_calls[0].id, &response.tool_calls[0].name, &response.tool_calls[0].arguments),
        ChatMessage::tool(&response.tool_calls[0].id, r#"{"temp": 28, "condition": "sunny"}"#),
    ]);
    let response2 = provider.chat(request2).await.unwrap();

    assert_eq!(response2.content, "Shanghai is sunny!");
    assert!(response2.tool_calls.is_empty());
}

// ── 6. Streaming tool-call ────────────────────────────────────────────────

#[tokio::test]
async fn stream_tool_call_full_assembly() {
    // Verify that collect_response correctly assembles streamed tool calls
    let server = MockServer::start().await;

    let mut sse_body = String::new();
    // Tool call start
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_full",
                    "function": {"name": "search", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Argument fragments
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"q\":"}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "\"rust\"}"}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Stop with tool_calls reason
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }).to_string()));
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = make_provider(&server.uri());
    let request = ChatRequest::new(vec![ChatMessage::user("search")]);
    let stream = provider.stream(request).await.unwrap();
    let response = stream.collect_response().await.unwrap();

    // Verify the assembled tool call has correct id, name, and concatenated arguments
    assert_eq!(response.tool_calls.len(), 1, "Expected exactly 1 assembled tool call");
    assert_eq!(response.tool_calls[0].id, "call_full");
    assert_eq!(response.tool_calls[0].name, "search");
    assert_eq!(response.tool_calls[0].arguments, r#"{"q":"rust"}"#, "Arguments should be concatenated from fragments");
}
