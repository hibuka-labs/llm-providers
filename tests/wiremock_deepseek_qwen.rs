//! Wiremock integration tests for DeepSeek and Qwen providers.
//!
//! Covers Phase 4b acceptance criteria from docs/adapter-design.md §15.5.2:
//! - DeepSeek streaming + non-streaming calls
//! - Qwen streaming + non-streaming calls
//! - Provider info() returns correct name
//! - create_provider() routes correctly

use llm_trait::{ChatMessage, ChatRequest, LlmProvider, StreamChunk};
use llm_unified::{DeepSeekProvider, GenericProvider, QwenProvider, OpenAiProtocol};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// OpenAI-format non-streaming response.
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

/// Build SSE data line.
fn sse_data(data: &str) -> String {
    format!("data: {data}\n\n")
}

/// Build a simple streaming SSE body.
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
    let provider = DeepSeekProvider::new("sk-test", "deepseek-chat", None);
    let info = provider.info();
    assert_eq!(info.name, "deepseek");
    assert_eq!(info.model, "deepseek-chat");
}

#[tokio::test]
async fn deepseek_chat_returns_response() {
    let server = MockServer::start().await;

    let body = openai_text_response("Hello from DeepSeek!");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = DeepSeekProvider::new("sk-test", "deepseek-chat", Some(&server.uri()));
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
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = DeepSeekProvider::new("sk-test", "deepseek-chat", Some(&server.uri()));
    let request = ChatRequest::new(vec![ChatMessage::user("test")]);
    let stream = provider.stream(request).await.unwrap();
    let text = stream.collect_text().await.unwrap();

    assert_eq!(text, "DeepSeek streams!");
}

// ── Qwen tests ───────────────────────────────────────────────────────────────

#[test]
fn qwen_provider_info() {
    let provider = QwenProvider::new("sk-test", "qwen-plus", None);
    let info = provider.info();
    assert_eq!(info.name, "qwen");
    assert_eq!(info.model, "qwen-plus");
}

#[tokio::test]
async fn qwen_chat_returns_response() {
    let server = MockServer::start().await;

    let body = openai_text_response("Hello from Qwen!");
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = QwenProvider::new("sk-test", "qwen-plus", Some(&server.uri()));
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
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let provider = QwenProvider::new("sk-test", "qwen-plus", Some(&server.uri()));
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
    let provider = llm_unified::create_provider(&config).unwrap();
    // DeepSeek uses OpenAI protocol, so provider name is "openai"
    assert_eq!(provider.info().name, "openai");
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
    let provider = llm_unified::create_provider(&config).unwrap();
    // Qwen uses OpenAI protocol, so provider name is "openai"
    assert_eq!(provider.info().name, "openai");
}

#[test]
fn factory_routes_openai_protocol() {
    let config = llm_trait::LlmConfig {
        protocol: Some("openai".to_string()),
        api_key: "sk-test".to_string(),
        model: "gpt-4o".to_string(),
        base_url: "https://custom.api.com/v1".to_string(),
        options: Default::default(),
    };
    let provider = llm_unified::create_provider(&config).unwrap();
    assert_eq!(provider.info().name, "openai");
}

// ── DeepSeek tool call tests ──────────────────────────────────────────────

#[tokio::test]
async fn deepseek_chat_parses_tool_calls() {
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
                    "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
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

    let provider = DeepSeekProvider::new("sk-test", "deepseek-chat", Some(&server.uri()));
    let request = ChatRequest::new(vec![ChatMessage::user("search for rust")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "search");
    assert_eq!(response.tool_calls[0].id, "call_1");
    assert_eq!(response.tool_calls[0].arguments, "{\"q\":\"rust\"}");
    assert_eq!(response.finish_reason, llm_trait::FinishReason::ToolCalls);
}

#[tokio::test]
async fn deepseek_stream_increments_tool_call() {
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
                    "id": "call_ds",
                    "function": {"name": "read_file", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Argument fragment
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"path\":\"/tmp\"}"}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Stop
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

    let provider = DeepSeekProvider::new("sk-test", "deepseek-chat", Some(&server.uri()));
    let request = ChatRequest::new(vec![ChatMessage::user("read file")]);
    let mut stream = provider.stream(request).await.unwrap();

    let mut tool_call_chunks = 0;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::ToolCall(_)) = chunk {
            tool_call_chunks += 1;
        }
    }
    assert_eq!(tool_call_chunks, 2, "Expected 2 tool call chunks (start + 1 arg fragment)");
}

// ── Qwen tool call tests ─────────────────────────────────────────────────

#[tokio::test]
async fn qwen_chat_parses_tool_calls() {
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
                    "id": "call_q1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{\"expr\":\"1+1\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
    });
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&body))
        .mount(&server)
        .await;

    let provider = QwenProvider::new("sk-test", "qwen-plus", Some(&server.uri()));
    let request = ChatRequest::new(vec![ChatMessage::user("calculate 1+1")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "calculate");
    assert_eq!(response.tool_calls[0].id, "call_q1");
    assert_eq!(response.tool_calls[0].arguments, "{\"expr\":\"1+1\"}");
    assert_eq!(response.finish_reason, llm_trait::FinishReason::ToolCalls);
}

#[tokio::test]
async fn qwen_stream_increments_tool_call() {
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
                    "id": "call_qw",
                    "function": {"name": "shell", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Argument fragment
    sse_body.push_str(&sse_data(&serde_json::json!({
        "id": "chatcmpl-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"cmd\":\"ls\"}"}
                }]
            },
            "finish_reason": null
        }]
    }).to_string()));
    // Stop
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

    let provider = QwenProvider::new("sk-test", "qwen-plus", Some(&server.uri()));
    let request = ChatRequest::new(vec![ChatMessage::user("run ls")]);
    let mut stream = provider.stream(request).await.unwrap();

    let mut tool_call_chunks = 0;
    while let Some(chunk) = stream.next().await {
        if let Ok(StreamChunk::ToolCall(_)) = chunk {
            tool_call_chunks += 1;
        }
    }
    assert_eq!(tool_call_chunks, 2, "Expected 2 tool call chunks (start + 1 arg fragment)");
}

#[tokio::test]
async fn qwen_chat_tool_call_multi_turn() {
    let server = MockServer::start().await;

    // Round 1: tool call
    let tool_call_body = serde_json::json!({
        "id": "chatcmpl-tc1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_q2",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{\"expr\":\"2*3\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    });

    // Round 2: final text
    let final_body = serde_json::json!({
        "id": "chatcmpl-tc2",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "The answer is 6."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}
    });

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

    let provider = QwenProvider::new("sk-test", "qwen-plus", Some(&server.uri()));

    // Round 1
    let request = ChatRequest::new(vec![ChatMessage::user("what is 2*3?")]);
    let response = provider.chat(request).await.unwrap();

    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "calculate");

    // Round 2: send tool result
    let request2 = ChatRequest::new(vec![
        ChatMessage::user("what is 2*3?"),
        ChatMessage::assistant_tool_call(&response.tool_calls[0].id, &response.tool_calls[0].name, &response.tool_calls[0].arguments),
        ChatMessage::tool(&response.tool_calls[0].id, "6"),
    ]);
    let response2 = provider.chat(request2).await.unwrap();

    assert_eq!(response2.content, "The answer is 6.");
    assert!(response2.tool_calls.is_empty());
}
