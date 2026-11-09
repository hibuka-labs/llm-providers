use super::*;
use crate::protocol::anthropic::types::AnthropicError;
use llm_trait::{
    ChatMessage, FinishReason, ImageAttachment, Protocol, ReasoningConfig, ReasoningMode,
    StreamChunk,
};

fn make_protocol() -> AnthropicProtocol {
    AnthropicProtocol::new("sk-test", "claude-sonnet", None)
}

fn make_protocol_with_profile(profile: ModelProfile) -> AnthropicProtocol {
    AnthropicProtocol::new("sk-test", "claude-sonnet", None).with_model_profile(profile)
}

fn claude_profile() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "anthropic",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    }
}

// ── Claude: thinking block IS sent ──

#[test]
fn claude_sends_thinking_block() {
    let proto = make_protocol_with_profile(claude_profile());
    let reasoning = ReasoningConfig {
        budget_tokens: Some(4096),
        ..Default::default()
    };
    let req = ChatRequest::new(vec![ChatMessage::user("think")]).with_reasoning(reasoning);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["thinking"]["type"], "enabled");
    assert_eq!(raw.body["thinking"]["budget_tokens"], 4096);
}

// ── Non-streaming reasoning_content ──

#[test]
fn parse_response_with_thinking_block() {
    let body = serde_json::json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me reason..."},
            {"type": "text", "text": "The answer is 42."}
        ],
        "model": "claude-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(serde_json::to_vec(&body).unwrap().as_slice())
            .unwrap();
    assert_eq!(resp.content, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me reason..."));
    // No signature in this response
    assert!(resp.thinking_signature.is_none());
}

#[test]
fn parse_response_with_thinking_block_and_signature() {
    let body = serde_json::json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me reason...", "signature": "sig_abc123"},
            {"type": "text", "text": "The answer is 42."}
        ],
        "model": "claude-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(serde_json::to_vec(&body).unwrap().as_slice())
            .unwrap();
    assert_eq!(resp.content, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me reason..."));
    // Signature should be extracted
    assert_eq!(resp.thinking_signature.as_deref(), Some("sig_abc123"));
}

// ── convert_messages ──

#[test]
fn convert_messages_basic() {
    let msgs = vec![ChatMessage::user("hi")];
    let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
    assert!(sys.is_none());
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "user");
}

/// A tool result that carried no picture keeps the bare string form —
/// the wire stays exactly as it was before the image face existed.
#[test]
fn a_word_only_tool_result_stays_a_plain_string() {
    let msgs = vec![ChatMessage::tool("tc1", "done")];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 1);
    let result = &out[0]["content"][0];
    assert_eq!(result["type"], "tool_result");
    assert_eq!(result["tool_use_id"], "tc1");
    assert_eq!(result["content"], "done");
}

/// A tool result that handed the model pictures puts them beside the
/// words: a block list of text + image, inside the same tool_result.
#[test]
fn a_tool_result_carries_its_pictures_beside_the_words() {
    let msgs = vec![ChatMessage::tool_with_images(
        "tc1",
        "image: the whole desktop",
        vec![ImageAttachment::Base64 {
            data: "aGVsbG8=".to_string(),
            media_type: Some("image/png".to_string()),
            detail: None,
        }],
    )];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    let result = &out[0]["content"][0];
    assert_eq!(result["type"], "tool_result");
    let blocks = result["content"].as_array().expect("block list");
    assert_eq!(blocks.len(), 2, "one text block + one image block");
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "image: the whole desktop");
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["type"], "base64");
    assert_eq!(blocks[1]["source"]["media_type"], "image/png");
    assert_eq!(blocks[1]["source"]["data"], "aGVsbG8=");
}

/// Two pictures keep their order and both arrive — a screenshot the
/// model must compare against the one before it.
#[test]
fn every_picture_a_tool_returned_arrives_in_order() {
    let attach = |d: &str| ImageAttachment::Base64 {
        data: d.to_string(),
        media_type: Some("image/png".to_string()),
        detail: None,
    };
    let msgs = vec![ChatMessage::tool_with_images(
        "tc1",
        "two shots",
        vec![attach("Zmlyc3Q="), attach("c2Vjb25k")],
    )];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    let blocks = out[0]["content"][0]["content"]
        .as_array()
        .expect("block list");
    let data: Vec<&str> = blocks
        .iter()
        .filter_map(|b| b["source"]["data"].as_str())
        .collect();
    assert_eq!(data, ["Zmlyc3Q=", "c2Vjb25k"]);
}

#[test]
fn convert_messages_with_system() {
    let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
    let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
    assert_eq!(sys.as_deref(), Some("sys"));
    assert_eq!(out.len(), 1);
}

#[test]
fn convert_messages_multiple_system_messages_merged() {
    // Multiple System messages should be merged, not just keep the last one.
    let msgs = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::system("Always respond in Chinese."),
        ChatMessage::user("hi"),
    ];
    let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
    let system = sys.expect("should have system prompt");
    assert!(
        system.contains("helpful assistant"),
        "first system message should be preserved"
    );
    assert!(
        system.contains("Chinese"),
        "second system message should be preserved"
    );
    assert_eq!(out.len(), 1);
}

#[test]
fn convert_messages_assistant_reasoning_content_preserved() {
    // convert_messages now preserves reasoning_content as a thinking block
    // when thinking_signature is present.
    let msgs = vec![
        ChatMessage::user("think about this"),
        ChatMessage::Assistant {
            content: Some("the answer".to_string()),
            reasoning_content: Some("let me reason step by step...".to_string()),
            thinking_signature: Some("sig_abc123".to_string()),
            tool_calls: None,
        },
        ChatMessage::user("follow up"),
    ];
    let (_sys, out) = AnthropicProtocol::convert_messages(&msgs);
    let assistant_msg = out.iter().find(|m| m["role"] == "assistant").unwrap();
    let blocks = assistant_msg["content"].as_array().unwrap();

    // Should have thinking block + text block
    assert_eq!(blocks.len(), 2);

    // First block should be thinking with signature
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "let me reason step by step...");
    assert_eq!(blocks[0]["signature"], "sig_abc123");

    // Second block should be text
    assert_eq!(blocks[1]["type"], "text");
    assert_eq!(blocks[1]["text"], "the answer");
}

#[test]
fn convert_messages_assistant_no_thinking_without_signature() {
    // When reasoning_content exists but thinking_signature is None,
    // no thinking block should be generated (signature is required by Anthropic)
    let msgs = vec![
        ChatMessage::user("think about this"),
        ChatMessage::assistant_with_reasoning("the answer", "let me reason..."),
        ChatMessage::user("follow up"),
    ];
    let (_sys, out) = AnthropicProtocol::convert_messages(&msgs);
    let assistant_msg = out.iter().find(|m| m["role"] == "assistant").unwrap();
    let blocks = assistant_msg["content"].as_array().unwrap();

    // Should only have text block (no thinking without signature)
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "text");
}

// ── build_request ──

#[test]
fn build_request_basic() {
    let proto = make_protocol();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Stream).unwrap();
    // URL should be {base_url}/v1/messages (Anthropic standard path)
    assert!(
        raw.url.ends_with("/v1/messages"),
        "URL should end with /v1/messages, got: {}",
        raw.url
    );
    assert!(raw.stream);
    assert_eq!(raw.method, HttpMethod::Post);
    assert!(raw.headers.contains_key("x-api-key"));
    assert!(raw.headers.contains_key("anthropic-version"));
    assert_eq!(raw.body["model"], "claude-sonnet");
}

#[test]
fn build_request_url_with_v1_suffix() {
    // When base_url already ends with /v1, don't double it
    let proto = AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1"));
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.url, "https://api.example.com/v1/messages");
}

#[test]
fn build_request_url_without_v1_suffix() {
    // When base_url doesn't end with /v1, add it
    let proto = AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com"));
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.url, "https://api.example.com/v1/messages");
}

#[test]
fn build_request_url_with_trailing_slash() {
    // Trailing slash should not cause double-slash in URL
    let proto =
        AnthropicProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1/"));
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert!(
        !raw.url.contains("//v1"),
        "URL should not have double-slash, got: {}",
        raw.url
    );
    assert!(
        raw.url.ends_with("/v1/messages"),
        "URL should end with /v1/messages, got: {}",
        raw.url
    );
}

// ── info ──

#[test]
fn convert_event_error_returns_err() {
    // Error events should return Err (stream-level error),
    // consistent with OpenAI's error handling.
    let event = MessagesStreamEvent::Error {
        error: AnthropicError {
            error_type: "overloaded_error".to_string(),
            message: "Too many requests".to_string(),
        },
    };
    let result = AnthropicProtocol::convert_event(event);
    assert!(result.is_err(), "Error event should return Err");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("overloaded_error"),
        "Error should contain type"
    );
    assert!(
        err_msg.contains("Too many requests"),
        "Error should contain message"
    );
}

#[test]
fn convert_event_message_start_maps_input_tokens() {
    // Real payload shape from DashScope's anthropic-compatible endpoint.
    let event: MessagesStreamEvent = serde_json::from_str(
        r#"{"message":{"model":"qwen3.8-flash","id":"msg_1","role":"assistant",
            "type":"message","content":[],
            "usage":{"input_tokens":2,"output_tokens":0}},"type":"message_start"}"#,
    )
    .unwrap();
    let chunks = AnthropicProtocol::convert_event(event).unwrap();
    let StreamChunk::Usage(u) = &chunks[0] else {
        panic!("expected Usage chunk, got {chunks:?}");
    };
    assert_eq!(u.prompt_tokens, Some(2));
    assert_eq!(u.completion_tokens, Some(0));
}

#[test]
fn convert_event_message_delta_maps_optional_input_tokens() {
    // message_delta always carries the final output count; input_tokens is
    // optional per protocol (real Anthropic omits it, DashScope includes
    // the full-context count). Both shapes must survive the mapping.
    let with_input: MessagesStreamEvent = serde_json::from_str(
        r#"{"delta":{"stop_reason":"end_turn"},"type":"message_delta",
            "usage":{"output_tokens":26,"input_tokens":63,
                     "cache_creation_input_tokens":0,"cache_read_input_tokens":0}}"#,
    )
    .unwrap();
    let chunks = AnthropicProtocol::convert_event(with_input).unwrap();
    let StreamChunk::Usage(u) = &chunks[0] else {
        panic!("expected Usage chunk, got {chunks:?}");
    };
    assert_eq!(
        u.prompt_tokens,
        Some(63),
        "endpoint-reported input survives"
    );
    assert_eq!(u.completion_tokens, Some(26));

    let without_input: MessagesStreamEvent = serde_json::from_str(
        r#"{"delta":{"stop_reason":"end_turn"},"type":"message_delta",
            "usage":{"output_tokens":26}}"#,
    )
    .unwrap();
    let chunks = AnthropicProtocol::convert_event(without_input).unwrap();
    let StreamChunk::Usage(u) = &chunks[0] else {
        panic!("expected Usage chunk, got {chunks:?}");
    };
    assert_eq!(
        u.prompt_tokens, None,
        "absent input stays None for the merger"
    );
}

#[test]
fn info_with_profile() {
    let proto = make_protocol_with_profile(claude_profile());
    let info = proto.info();
    assert_eq!(info.name, "anthropic");
}

#[test]
fn info_without_profile() {
    let proto = make_protocol();
    let info = proto.info();
    assert_eq!(info.name, "anthropic");
}

#[test]
fn build_request_budget_equals_max_tokens_must_increase() {
    // Anthropic requires max_tokens > budget_tokens.
    // When budget == max_tokens, we must increase max_tokens.
    let profile = ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "anthropic",
        capabilities: Capabilities {
            max_output_tokens: Some(8_192),
            ..claude_profile().capabilities
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    };
    let proto = make_protocol_with_profile(profile);
    let reasoning = ReasoningConfig {
        budget_tokens: Some(8_192), // == max_tokens
        ..Default::default()
    };
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    let max_tokens = raw.body["max_tokens"].as_u64().unwrap();
    let budget = raw.body["thinking"]["budget_tokens"].as_u64().unwrap();

    assert!(
        max_tokens > budget,
        "max_tokens ({max_tokens}) must be > budget_tokens ({budget}), Anthropic rejects equal values"
    );
}

#[test]
fn build_request_uses_profile_max_output_tokens() {
    // AnthropicProtocol now uses profile.max_output_tokens.
    let profile = ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "anthropic",
        capabilities: Capabilities {
            max_output_tokens: Some(64_000),
            ..claude_profile().capabilities
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    };
    let proto = make_protocol_with_profile(profile);
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    let max_tokens = raw.body["max_tokens"].as_u64().unwrap();

    assert_eq!(
        max_tokens, 64_000,
        "Anthropic should use profile.max_output_tokens"
    );
}

// ── convert_tools ──

#[test]
fn convert_tools_basic() {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
    })];
    let result = AnthropicProtocol::convert_tools(&tools);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["name"], "search");
    assert_eq!(result[0]["description"], "Search the web");
    assert_eq!(result[0]["input_schema"]["type"], "object");
}

#[test]
fn convert_tools_missing_description_and_params() {
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": { "name": "minimal" }
    })];
    let result = AnthropicProtocol::convert_tools(&tools);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["description"], "");
    assert_eq!(result[0]["input_schema"]["type"], "object");
}

#[test]
fn convert_tools_missing_function_key() {
    let tools = vec![serde_json::json!({"type": "function"})];
    let result = AnthropicProtocol::convert_tools(&tools);
    assert_eq!(result.len(), 0);
}

#[test]
fn convert_tools_missing_name() {
    let tools = vec![serde_json::json!({
        "function": { "description": "no name" }
    })];
    let result = AnthropicProtocol::convert_tools(&tools);
    assert_eq!(result.len(), 0);
}

// ── parse_anthropic_response: error paths ──

#[test]
fn parse_response_invalid_json() {
    let result = AnthropicProtocol::parse_anthropic_response(b"not json");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Failed to parse"));
}

#[test]
fn parse_response_error_object() {
    let body = serde_json::json!({
        "error": {"message": "rate limited", "type": "rate_error"}
    });
    let result = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("rate_error: rate limited")
    );
}

#[test]
fn parse_response_error_missing_fields() {
    let body = serde_json::json!({"error": {}});
    let result = AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("api_error: unknown error")
    );
}

// ── parse_anthropic_response: stop_reason variants ──

#[test]
fn parse_response_stop_reason_max_tokens() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "truncated"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Length);
}

#[test]
fn parse_response_stop_reason_stop_sequence() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "stop_sequence",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Stop);
}

#[test]
fn parse_response_stop_reason_refusal() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "refused"}],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::ContentFilter);
}

#[test]
fn parse_response_stop_reason_unknown() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "something_new",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(
        resp.finish_reason,
        FinishReason::Other("something_new".into())
    );
}

#[test]
fn parse_response_no_stop_reason_defaults_to_stop() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "x"}],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Stop);
}

// ── parse_anthropic_response: missing content/usage ──

#[test]
fn parse_response_no_content_array() {
    let body = serde_json::json!({
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.content, "");
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn parse_response_no_usage() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "x"}]
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.usage.prompt_tokens, None);
}

// ── parse_anthropic_response: tool_use with missing fields ──

#[test]
fn parse_response_tool_use_missing_fields() {
    let body = serde_json::json!({
        "content": [
            {"type": "tool_use"},
            {"type": "tool_use", "id": "t1"},
            {"type": "tool_use", "id": "t2", "name": "fn"},
            {"type": "tool_use", "id": "t3", "name": "fn2", "input": {"x": 1}}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.tool_calls.len(), 4);
    assert_eq!(resp.tool_calls[0].id, "");
    assert_eq!(resp.tool_calls[0].name, "");
    assert_eq!(resp.tool_calls[0].arguments, "{}");
    assert_eq!(resp.tool_calls[1].name, "");
    assert_eq!(resp.tool_calls[2].arguments, "{}");
    assert_eq!(resp.tool_calls[3].arguments, r#"{"x":1}"#);
}

// ── parse_anthropic_response: thinking block ──

#[test]
fn parse_response_thinking_block_with_signature() {
    let body = serde_json::json!({
        "content": [
            {"type": "thinking", "thinking": "let me think...", "signature": "sig123"},
            {"type": "text", "text": "answer"}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.reasoning_content.as_deref(), Some("let me think..."));
    assert_eq!(resp.thinking_signature.as_deref(), Some("sig123"));
    assert_eq!(resp.content, "answer");
}

#[test]
fn parse_response_thinking_block_no_signature() {
    let body = serde_json::json!({
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "ok"}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let resp =
        AnthropicProtocol::parse_anthropic_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.reasoning_content.as_deref(), Some("hmm"));
    assert_eq!(resp.thinking_signature, None);
}

// ── convert_messages: tool message ──

#[test]
fn convert_messages_tool_role() {
    let msgs = vec![ChatMessage::tool("call_1", "result data")];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "user");
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "tool_result");
    assert_eq!(content[0]["tool_use_id"], "call_1");
    assert_eq!(content[0]["content"], "result data");
}

// ── convert_messages: custom filtered ──

#[test]
fn convert_messages_custom_filtered() {
    let msgs = vec![
        ChatMessage::user("before"),
        ChatMessage::Custom {
            role: "custom".into(),
            data: serde_json::json!({"secret": true}),
        },
        ChatMessage::user("after"),
    ];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    // Custom filtered, two users merged
    assert_eq!(out.len(), 1);
    let content = out[0]["content"].as_array().unwrap();
    let text: String = content
        .iter()
        .map(|c| c["text"].as_str().unwrap_or(""))
        .collect();
    assert!(!text.contains("secret"));
}

// ── convert_messages: assistant with empty content ──

#[test]
fn convert_messages_assistant_empty_content() {
    let msgs = vec![ChatMessage::Assistant {
        content: None,
        reasoning_content: None,
        tool_calls: None,
        thinking_signature: None,
    }];
    let (_, out) = AnthropicProtocol::convert_messages(&msgs);
    // Empty assistant produces empty blocks array, filtered out by `!blocks.is_empty()`
    assert_eq!(out.len(), 0);
}

// ── capabilities ──

#[test]
fn capabilities_with_profile() {
    let profile = ModelProfile {
        protocol: Protocol::Anthropic,
        provider_name: "anthropic",
        capabilities: Capabilities {
            supports_vision: true,
            supports_thinking: true,
            ..Default::default()
        },
        reasoning_mode: ReasoningMode::Thinking,
        supported_extra_params: &[],
    };
    let proto = make_protocol_with_profile(profile);
    let caps = proto.capabilities();
    assert!(caps.supports_vision);
    assert!(caps.supports_thinking);
}

#[test]
fn capabilities_without_profile() {
    let proto = make_protocol();
    let caps = proto.capabilities();
    // Default Capabilities has supports_vision=true for Anthropic
    assert!(caps.supports_vision);
}

// ── from_config ──

#[test]
fn from_config_with_string_max_tokens() {
    use llm_trait::LlmConfig;
    let mut options = std::collections::HashMap::new();
    options.insert("max_tokens".into(), serde_json::json!("4096"));
    let config = LlmConfig {
        api_key: "sk-test".into(),
        model: "claude-sonnet".into(),
        base_url: "https://api.anthropic.com".into(),
        options,
        ..Default::default()
    };
    let proto = AnthropicProtocol::from_config(&config);
    assert_eq!(proto.max_tokens, 4096);
}

#[test]
fn from_config_with_numeric_max_tokens() {
    use llm_trait::LlmConfig;
    let mut options = std::collections::HashMap::new();
    options.insert("max_tokens".into(), serde_json::json!(16384));
    let config = LlmConfig {
        api_key: "sk-test".into(),
        model: "claude-sonnet".into(),
        base_url: "https://api.anthropic.com".into(),
        options,
        ..Default::default()
    };
    let proto = AnthropicProtocol::from_config(&config);
    assert_eq!(proto.max_tokens, 16384);
}

#[test]
fn from_config_default_max_tokens() {
    use llm_trait::LlmConfig;
    let config = LlmConfig {
        api_key: "sk-test".into(),
        model: "claude-sonnet".into(),
        base_url: "https://api.anthropic.com".into(),
        ..Default::default()
    };
    let proto = AnthropicProtocol::from_config(&config);
    assert_eq!(proto.max_tokens, 16384);
}

// ── build_request: system message ──

#[test]
fn build_request_with_system_message() {
    let proto = make_protocol();
    let req = ChatRequest::new(vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("hello"),
    ]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["system"], "You are a helpful assistant.");
}

// ── build_request: tools ──

#[test]
fn build_request_with_tools() {
    let proto = make_protocol();
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
    })];
    let req = ChatRequest::new(vec![ChatMessage::user("search for cats")]).with_tools(tools);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert!(raw.body["tools"].is_array());
    assert_eq!(raw.body["tools"].as_array().unwrap().len(), 1);
    assert_eq!(raw.body["tool_choice"]["type"], "auto");
}

// ── proptest: convert_messages ──

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a random ChatMessage (simplified: system, user, assistant, tool).
    fn arb_chat_message() -> impl Strategy<Value = ChatMessage> {
        prop_oneof![
            "[a-z ]{0,50}".prop_map(|s| ChatMessage::system(&s)),
            "[a-z ]{0,50}".prop_map(|s| ChatMessage::user(&s)),
            "[a-z ]{0,50}".prop_map(|s| ChatMessage::assistant(&s)),
            ("[a-z]{1,10}", "[a-z ]{0,50}")
                .prop_map(|(id, content)| ChatMessage::tool(&id, &content)),
        ]
    }

    proptest! {
        #[test]
        fn convert_messages_output_len_le_input_len(
            messages in prop::collection::vec(arb_chat_message(), 0..20)
        ) {
            let (_, out) = AnthropicProtocol::convert_messages(&messages);
            // System messages extracted → output ≤ non-system count
            let non_system_count = messages.iter()
                .filter(|m| !matches!(m, ChatMessage::System { .. }))
                .filter(|m| !matches!(m, ChatMessage::Custom { .. }))
                .count();
            assert!(out.len() <= non_system_count,
                "output {} > non-system input {}", out.len(), non_system_count);
        }

        #[test]
        fn convert_messages_output_roles_are_valid(
            messages in prop::collection::vec(arb_chat_message(), 0..20)
        ) {
            let (_, out) = AnthropicProtocol::convert_messages(&messages);
            let valid_roles = ["user", "assistant"];
            for msg in &out {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                assert!(valid_roles.contains(&role),
                    "invalid role: {:?}, full message: {:?}", role, msg);
            }
        }

        #[test]
        fn convert_messages_system_extracted(
            content in "[a-z ]{1,50}"
        ) {
            let msgs = vec![ChatMessage::system(&content)];
            let (sys, out) = AnthropicProtocol::convert_messages(&msgs);
            assert!(sys.is_some(), "system message should be extracted");
            assert!(out.is_empty(), "system message should not appear in output");
        }
    }
}
