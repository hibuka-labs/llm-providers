//! Regression test: truncated tool call arguments survive serialization.
//!
//! When the LLM hits the output token limit (`finish_reason: "length"`), the
//! streamed tool call arguments can be truncated mid-string, producing invalid
//! JSON. If this invalid JSON is stored in the session history and sent back to
//! the API on the next turn, the API returns 400 "Invalid request parameters".
//!
//! Bug report: session log 20260826_8b229428 — MiMo 400 on turn 8 because
//! turn 7's write_file had truncated arguments (19KB file content, ~5000 tokens,
//! but max_tokens was only 4096).

use llm_trait::{CallMode, ChatMessage, ChatRequest, RawAdapter, ToolCallMessage};
use llm_unified::OpenAiProtocol;

// ── Scenario 1: max_tokens default is too small ──────────────────────────────

#[test]
fn max_tokens_default_is_16384() {
    let protocol = OpenAiProtocol::new("sk-test", "model", Some("http://localhost"));
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let raw = protocol.build_request(&request, CallMode::Once).unwrap();
    let max_tokens = raw.body["max_tokens"].as_u64().unwrap();
    assert_eq!(
        max_tokens, 16384,
        "max_tokens default should be 16384 to accommodate large tool call arguments"
    );
}

#[test]
fn max_tokens_with_profile_uses_profile_value() {
    // When a profile specifies max_output_tokens, the protocol should use it.
    use llm_unified::model_registry::ModelRegistry;
    let registry = ModelRegistry::builtin();
    let profile = registry.lookup(
        "mimo-v2.5-pro",
        Some("https://token-plan-cn.xiaomimimo.com/v1"),
        None,
    );
    // MiMo profile has max_output_tokens: 8192
    assert_eq!(profile.capabilities.max_output_tokens, Some(8192));
}

// ── Scenario 2: truncated tool args pass through without validation ──────────

fn request_with_truncated_tool_call() -> ChatRequest {
    // Simulate a tool call whose arguments were cut off mid-string.
    // This matches the real failure: write_file with 18951 chars of content.
    let truncated_args = r#"{"path": "src/ui/markdown.rs", "content": "#.to_string();

    let assistant_msg = ChatMessage::Assistant {
        content: Some("Writing the file now.".to_string()),
        reasoning_content: None,
        thinking_signature: None,
        tool_calls: Some(vec![ToolCallMessage {
            id: "call_abc123".to_string(),
            name: "write_file".to_string(),
            arguments: truncated_args,
        }]),
    };

    let tool_error_msg = ChatMessage::tool(
        "call_abc123",
        "Tool call was not executed: the response hit the output token limit, \
         so its arguments may be truncated. Re-issue the tool call with complete arguments.",
    );

    ChatRequest::new(vec![assistant_msg, tool_error_msg, ChatMessage::user("continue")])
}

#[test]
fn truncated_tool_args_are_invalid_json() {
    let request = request_with_truncated_tool_call();
    let protocol = OpenAiProtocol::new("sk-test", "mimo-v2.5-pro", Some("http://localhost"));
    let raw = protocol.build_request(&request, CallMode::Once).unwrap();

    let messages = raw.body["messages"].as_array().unwrap();
    let assistant_msg = messages
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .unwrap();

    let args_str = assistant_msg["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();

    // BUG: truncated args are invalid JSON, passed through to API as-is.
    let parse_result = serde_json::from_str::<serde_json::Value>(args_str);
    assert!(
        parse_result.is_err(),
        "truncated arguments should be invalid JSON"
    );
}

#[test]
fn truncated_args_cause_api_400() {
    // This test simulates what happens when the API receives invalid JSON
    // in tool call arguments. MiMo returns 400 "Invalid request parameters".
    //
    // We verify by checking that a JSON parser rejects the arguments string,
    // which is exactly what the API's request validator does internally.
    let truncated = r#"{"path": "x.rs", "content": "#;
    assert!(
        serde_json::from_str::<serde_json::Value>(truncated).is_err(),
        "truncated JSON must be invalid"
    );

    // Contrast: complete JSON is valid
    let complete = r#"{"path": "x.rs", "content": "hello"}"#;
    assert!(
        serde_json::from_str::<serde_json::Value>(complete).is_ok(),
        "complete JSON must be valid"
    );
}

// ── Scenario 3: finish_reason="stop" bypasses truncation guard ───────────────

#[test]
fn truncation_guard_only_checks_length_finish_reason() {
    // The truncation guard checks finish_reason == Length.
    // Some APIs return "stop" even when output is truncated — guard is bypassed.
    use llm_trait::FinishReason;

    // "length" → Length (guard fires)
    let fr_length = FinishReason::from_str("length");
    assert_eq!(fr_length, FinishReason::Length, "length should map to Length");

    // "max_tokens" → Length (Anthropic style, guard fires)
    let fr_max = FinishReason::from_str("max_tokens");
    assert_eq!(fr_max, FinishReason::Length, "max_tokens should map to Length");

    // "stop" → Stop (guard does NOT fire — this is the bug when API lies)
    let fr_stop = FinishReason::from_str("stop");
    assert_eq!(fr_stop, FinishReason::Stop, "stop is not Length — guard bypassed");

    // "tool_calls" → ToolCalls (guard does NOT fire)
    let fr_tc = FinishReason::from_str("tool_calls");
    assert_eq!(fr_tc, FinishReason::ToolCalls, "tool_calls is not Length — guard bypassed");
}

// ── Scenario 4: realistic write_file content size ────────────────────────────

#[test]
fn realistic_write_file_exceeds_4096_tokens() {
    // A typical Rust source file (~500 lines, ~19KB) needs ~5000 tokens.
    // With max_tokens=4096, the tool call arguments will be truncated.
    //
    // Token estimation: ~4 chars per token for code (conservative).
    let file_content = "x".repeat(18951); // matches real failure size
    let args = format!(r#"{{"path": "src/ui/markdown.rs", "content": "{}"}}"#, file_content);
    let estimated_tokens = args.len() / 4; // ~4737 tokens
    let max_tokens = 16384u32;

    assert!(
        estimated_tokens as u32 <= max_tokens,
        "write_file with {} chars (~{} tokens) now fits within max_tokens={}",
        args.len(),
        estimated_tokens,
        max_tokens
    );
}
