use super::*;
use llm_trait::{
    ChatMessage, FinishReason, Protocol, ReasoningConfig, ReasoningEffort, ReasoningMode,
};

fn make_protocol() -> OpenAiProtocol {
    OpenAiProtocol::new("sk-test", "gpt-4o", None)
}

fn make_protocol_with_profile(profile: ModelProfile) -> OpenAiProtocol {
    OpenAiProtocol::new("sk-test", "gpt-4o", None).with_model_profile(profile)
}

fn mimo_openai_profile() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "mimo",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
            ..Default::default()
        },
        reasoning_mode: ReasoningMode::None,
        supported_extra_params: &[],
    }
}

fn deepseek_profile() -> ModelProfile {
    ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "deepseek",
        capabilities: Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_thinking: true,
            max_context_tokens: Some(1_000_000),
            max_output_tokens: Some(16_384),
            ..Default::default()
        },
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &["reasoning_effort"],
    }
}

// ── MiMo: no reasoning_effort sent ──

#[test]
fn mimo_no_reasoning_effort() {
    let proto = make_protocol_with_profile(mimo_openai_profile());
    let reasoning = ReasoningConfig {
        effort: Some(ReasoningEffort::Medium),
        ..Default::default()
    };
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    // MiMo's reasoning_mode is None, so reasoning_effort should NOT be sent
    assert!(
        raw.body.get("reasoning_effort").is_none(),
        "MiMo should not send reasoning_effort"
    );
}

// ── DeepSeek: reasoning_effort IS sent ──

#[test]
fn deepseek_sends_reasoning_effort() {
    let proto = make_protocol_with_profile(deepseek_profile());
    let reasoning = ReasoningConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    };
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["reasoning_effort"], "high");
}

// ── Non-streaming reasoning_content extraction ──

#[test]
fn parse_response_with_reasoning_content() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think step by step..."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    });
    let resp = OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();
    assert_eq!(resp.content, "The answer is 42.");
    assert_eq!(
        resp.reasoning_content.as_deref(),
        Some("Let me think step by step...")
    );
}

#[test]
fn parse_response_without_reasoning_content() {
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let resp = OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();
    assert!(resp.reasoning_content.is_none());
}

// ── convert_messages ──

#[test]
fn convert_messages_basic() {
    let msgs = vec![ChatMessage::user("hi")];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "user");
    assert_eq!(out[0]["content"], "hi");
}

#[test]
fn convert_messages_with_system() {
    let msgs = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["role"], "system");
    assert_eq!(out[1]["role"], "user");
}

#[test]
fn convert_messages_consecutive_assistant_must_merge() {
    // Two assistant messages: one with text, one with tool_calls.
    // These should NOT be merged (tool_calls needs null content).
    let msgs = vec![
        ChatMessage::assistant("Let me check."),
        ChatMessage::assistant_tool_call("call_1", "search", r#"{"q":"test"}"#),
        ChatMessage::tool("call_1", "result"),
    ];
    let out = OpenAiProtocol::convert_messages(&msgs);
    // Should be 3 messages: assistant(text), assistant(tool_calls), tool
    // because text+tool_calls should not be merged
    assert_eq!(out.len(), 3);
    assert_eq!(out[0]["role"], "assistant");
    assert_eq!(out[0]["content"], "Let me check.");
    assert_eq!(out[1]["role"], "assistant");
    assert!(out[1].get("tool_calls").is_some());
    assert_eq!(out[2]["role"], "tool");
}

#[test]
fn convert_messages_consecutive_user_must_merge() {
    let msgs = vec![
        ChatMessage::user("first question"),
        ChatMessage::user("follow up"),
    ];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "user");
}

#[test]
fn convert_messages_consecutive_user_with_images_preserves_images() {
    // Consecutive user messages with images now merge as arrays.
    use llm_trait::ImageAttachment;
    let msgs = vec![
        ChatMessage::user_with_images(
            "describe this image",
            vec![ImageAttachment::Url {
                url: "https://example.com/cat.jpg".to_string(),
                detail: None,
            }],
        ),
        ChatMessage::user("what color is it?"),
    ];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out.len(), 1);
    let merged = &out[0];
    assert_eq!(merged["role"], "user");

    // Content should be an array (merged parts)
    let content = merged.get("content").unwrap();
    assert!(content.is_array(), "Content should be an array after merge");
    let parts = content.as_array().unwrap();

    // Should have: text("describe this image"), image_url, text("what color is it?")
    assert_eq!(parts.len(), 3, "Expected 3 parts: text, image, text");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "describe this image");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.jpg");
    assert_eq!(parts[2]["type"], "text");
    assert_eq!(parts[2]["text"], "what color is it?");
}

#[test]
fn convert_messages_custom_filtered_out() {
    // ChatMessage::Custom should be filtered out, not forwarded to LLM.
    let msgs = vec![
        ChatMessage::Custom {
            role: "metadata".to_string(),
            data: serde_json::json!({"secret": "should_not_leak"}),
        },
        ChatMessage::user("hello"),
    ];
    let out = OpenAiProtocol::convert_messages(&msgs);
    // Custom message should be completely dropped
    assert_eq!(out.len(), 1, "Custom message should be filtered out");
    let content = out[0]["content"].as_str().unwrap();
    assert!(
        !content.contains("should_not_leak"),
        "Custom message data should not appear in output, got: {}",
        content
    );
}

#[test]
fn convert_messages_assistant_tool_call_not_merged_with_text() {
    // Merging an assistant+tool_calls message with an assistant+text message
    // produces an invalid request. They should not be merged.
    let msgs = vec![
        ChatMessage::assistant_tool_call("call_1", "shell", r#"{"cmd":"ls"}"#),
        ChatMessage::assistant("Here are the results: ..."),
    ];
    let out = OpenAiProtocol::convert_messages(&msgs);
    // Should be 2 separate messages, not merged
    assert_eq!(
        out.len(),
        2,
        "assistant+tool_calls and assistant+text should not be merged"
    );
    // First should have tool_calls, no text content
    assert!(out[0].get("tool_calls").is_some());
    // Second should have text, no tool_calls
    assert_eq!(out[1]["content"], "Here are the results: ...");
}

// ── build_request ──

#[test]
fn build_request_basic() {
    let proto = make_protocol();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Stream).unwrap();
    assert!(raw.url.contains("/chat/completions"));
    assert!(raw.stream);
    assert_eq!(raw.method, HttpMethod::Post);
    assert_eq!(raw.body["model"], "gpt-4o");
}

#[test]
fn build_request_url_with_trailing_slash() {
    // Trailing slash should not cause double-slash in URL
    let proto = OpenAiProtocol::new("sk-test", "test-model", Some("https://api.example.com/v1/"));
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert!(
        !raw.url.contains("//v1"),
        "URL should not have double-slash, got: {}",
        raw.url
    );
    assert!(
        raw.url.ends_with("/v1/chat/completions"),
        "URL should end with /v1/chat/completions, got: {}",
        raw.url
    );
}

#[test]
fn build_request_stream_includes_usage_option() {
    // OpenAI streaming should include stream_options.include_usage
    // so that usage data is returned in the stream.
    let proto = make_protocol();
    let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let raw = proto.build_request(&req, CallMode::Stream).unwrap();
    assert!(
        raw.body["stream_options"]["include_usage"].as_bool() == Some(true),
        "stream requests should include stream_options.include_usage=true"
    );
}

// ── convert_delta ──

#[test]
fn convert_delta_text() {
    let delta = serde_json::json!({"content": "hello"});
    let chunks = OpenAiProtocol::convert_delta(&delta, 0);
    assert_eq!(chunks.len(), 1);
    assert!(matches!(&chunks[0], StreamChunk::Text(t) if t == "hello"));
}

#[test]
fn convert_delta_reasoning_content() {
    let delta = serde_json::json!({"reasoning_content": "thinking..."});
    let chunks = OpenAiProtocol::convert_delta(&delta, 0);
    assert_eq!(chunks.len(), 1);
    assert!(matches!(&chunks[0], StreamChunk::Thought(t) if t == "thinking..."));
}

// ── info ──

#[test]
fn info_with_profile() {
    let proto = make_protocol_with_profile(deepseek_profile());
    let info = proto.info();
    assert_eq!(info.name, "deepseek");
    assert_eq!(info.model, "gpt-4o");
}

#[test]
fn info_without_profile() {
    let proto = make_protocol();
    let info = proto.info();
    assert_eq!(info.name, "openai");
}

#[test]
fn parse_response_content_filter_finish_reason() {
    // content_filter should map to FinishReason::ContentFilter, not Other
    let body = serde_json::json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "content_filter"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let resp = OpenAiProtocol::parse_openai_response(serde_json::to_vec(&body).unwrap().as_slice())
        .unwrap();
    assert_eq!(
        resp.finish_reason,
        FinishReason::ContentFilter,
        "content_filter should map to ContentFilter, not Other"
    );
}

// ── parse_openai_response: error paths ──

#[test]
fn parse_response_invalid_json() {
    let result = OpenAiProtocol::parse_openai_response(b"not json");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Failed to parse"));
}

#[test]
fn parse_response_error_object() {
    let body = serde_json::json!({
        "error": {"message": "rate limited", "type": "rate_error"}
    });
    let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
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
    let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("api_error: unknown error")
    );
}

#[test]
fn parse_response_no_choices() {
    let body = serde_json::json!({"usage": {}});
    let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No choices"));
}

#[test]
fn parse_response_no_message() {
    let body = serde_json::json!({"choices": [{}]});
    let result = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No message"));
}

// ── parse_openai_response: finish_reason variants ──

#[test]
fn parse_response_finish_reason_length() {
    let body = serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Length);
}

#[test]
fn parse_response_finish_reason_unknown() {
    let body = serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "something_new"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(
        resp.finish_reason,
        FinishReason::Other("something_new".into())
    );
}

#[test]
fn parse_response_no_finish_reason_defaults_to_stop() {
    let body = serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "x"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.finish_reason, FinishReason::Stop);
}

// ── parse_openai_response: missing usage/tool_calls ──

#[test]
fn parse_response_no_usage() {
    let body = serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    assert_eq!(resp.usage.prompt_tokens, None);
}

#[test]
fn parse_response_tool_calls_missing_fields() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "fn", "arguments": "{}"}},
                    {"function": {"name": "fn2"}},
                    {"id": "t3"},
                    {}
                ]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    // filter_map skips entries missing id or function or name
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].id, "t1");
}

#[test]
fn parse_response_reasoning_content_empty() {
    let body = serde_json::json!({
        "choices": [{
            "message": {"role": "assistant", "content": "answer", "reasoning_content": ""},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let resp = OpenAiProtocol::parse_openai_response(&serde_json::to_vec(&body).unwrap()).unwrap();
    // Empty reasoning_content is filtered out
    assert_eq!(resp.reasoning_content, None);
}

// ── capabilities ──

#[test]
fn capabilities_with_profile() {
    let profile = ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "openai",
        capabilities: Capabilities {
            supports_vision: false,
            supports_thinking: true,
            ..Default::default()
        },
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &[],
    };
    let proto = make_protocol().with_model_profile(profile);
    let caps = proto.capabilities();
    assert!(!caps.supports_vision);
    assert!(caps.supports_thinking);
}

#[test]
fn capabilities_without_profile() {
    let proto = make_protocol();
    let caps = proto.capabilities();
    assert!(caps.supports_streaming);
    assert!(caps.supports_tools);
    assert!(caps.supports_vision);
    assert!(!caps.supports_thinking);
}

// ── from_config ──

#[test]
fn from_config_with_string_max_tokens() {
    let mut options = std::collections::HashMap::new();
    options.insert("max_tokens".into(), serde_json::json!("8192"));
    let config = LlmConfig {
        api_key: "sk-test".into(),
        model: "gpt-4".into(),
        base_url: "https://api.openai.com/v1".into(),
        options,
        ..Default::default()
    };
    let proto = OpenAiProtocol::from_config(&config);
    assert_eq!(proto.max_tokens, 8192);
}

#[test]
fn from_config_default_max_tokens() {
    let config = LlmConfig {
        api_key: "sk-test".into(),
        model: "gpt-4".into(),
        base_url: "https://api.openai.com/v1".into(),
        ..Default::default()
    };
    let proto = OpenAiProtocol::from_config(&config);
    assert_eq!(proto.max_tokens, 16384);
}

// ── reasoning effort arms ──

#[test]
fn build_request_reasoning_low() {
    let profile = ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "openai",
        capabilities: Capabilities::default(),
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &[],
    };
    let proto = make_protocol().with_model_profile(profile);
    let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_reasoning(ReasoningConfig {
        effort: Some(ReasoningEffort::Low),
        ..Default::default()
    });
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["reasoning_effort"], "low");
}

#[test]
fn build_request_reasoning_medium() {
    let profile = ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "openai",
        capabilities: Capabilities::default(),
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &[],
    };
    let proto = make_protocol().with_model_profile(profile);
    let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_reasoning(ReasoningConfig {
        effort: Some(ReasoningEffort::Medium),
        ..Default::default()
    });
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["reasoning_effort"], "medium");
}

#[test]
fn build_request_reasoning_xhigh() {
    let profile = ModelProfile {
        protocol: Protocol::OpenAi,
        provider_name: "openai",
        capabilities: Capabilities::default(),
        reasoning_mode: ReasoningMode::Effort,
        supported_extra_params: &[],
    };
    let proto = make_protocol().with_model_profile(profile);
    let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_reasoning(ReasoningConfig {
        effort: Some(ReasoningEffort::XHigh),
        ..Default::default()
    });
    let raw = proto.build_request(&req, CallMode::Once).unwrap();
    assert_eq!(raw.body["reasoning_effort"], "high");
}

// ── convert_messages: images ──

#[test]
fn convert_messages_base64_image() {
    use llm_trait::ImageAttachment;
    let msgs = vec![ChatMessage::user_with_images(
        "describe",
        vec![ImageAttachment::Base64 {
            data: "abc123".into(),
            media_type: Some("image/png".into()),
            detail: None,
        }],
    )];
    let out = OpenAiProtocol::convert_messages(&msgs);
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    assert!(
        content[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .contains("data:image/png;base64,abc123")
    );
}

#[test]
fn convert_messages_url_image_with_detail() {
    use llm_trait::ImageAttachment;
    let msgs = vec![ChatMessage::user_with_images(
        "describe",
        vec![ImageAttachment::Url {
            url: "https://example.com/img.jpg".into(),
            detail: Some(llm_trait::ImageDetail::High),
        }],
    )];
    let out = OpenAiProtocol::convert_messages(&msgs);
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content[1]["image_url"]["detail"], "high");
}

/// A `role:"tool"` message is string-only here, so a picture cannot ride
/// it — the model is told it missed one rather than being left silent.
#[test]
fn a_tool_result_reports_the_pictures_it_cannot_deliver() {
    use llm_trait::ImageAttachment;
    let msgs = vec![ChatMessage::tool_with_images(
        "tc1",
        "image: the whole desktop",
        vec![
            ImageAttachment::Base64 {
                data: "abc123".into(),
                media_type: Some("image/png".into()),
                detail: None,
            },
            ImageAttachment::Base64 {
                data: "abc123".into(),
                media_type: Some("image/png".into()),
                detail: None,
            },
        ],
    )];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out[0]["role"], "tool");
    let body = out[0]["content"].as_str().expect("string content");
    assert!(
        body.starts_with("image: the whole desktop"),
        "the words still lead: {body}"
    );
    assert!(
        body.contains("2 image(s)"),
        "the drop is named, never silent: {body}"
    );
}

/// Without pictures the wire is unchanged — no note on an ordinary result.
#[test]
fn a_word_only_tool_result_carries_no_note() {
    let msgs = vec![ChatMessage::tool("tc1", "done")];
    let out = OpenAiProtocol::convert_messages(&msgs);
    assert_eq!(out[0]["content"], "done");
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
            let out = OpenAiProtocol::convert_messages(&messages);
            // Merging can only reduce or maintain count; Custom messages are filtered
            let non_custom_count = messages.iter().filter(|m| !matches!(m, ChatMessage::Custom { .. })).count();
            assert!(out.len() <= non_custom_count,
                "output {} > non-custom input {}", out.len(), non_custom_count);
        }

        #[test]
        fn convert_messages_output_roles_are_valid(
            messages in prop::collection::vec(arb_chat_message(), 0..20)
        ) {
            let out = OpenAiProtocol::convert_messages(&messages);
            let valid_roles = ["system", "user", "assistant", "tool"];
            for msg in &out {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
                assert!(valid_roles.contains(&role),
                    "invalid role: {:?}, full message: {:?}", role, msg);
            }
        }

        #[test]
        fn convert_messages_single_user_preserved(
            content in "[a-z ]{1,50}"
        ) {
            let msgs = vec![ChatMessage::user(&content)];
            let out = OpenAiProtocol::convert_messages(&msgs);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0]["role"], "user");
            assert_eq!(out[0]["content"], content);
        }
    }
}
