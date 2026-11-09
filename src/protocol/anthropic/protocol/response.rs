use serde_json::Value;

use llm_trait::{ChatResponse, FinishReason, LlmError, ToolCall, UsageInfo};

use super::AnthropicProtocol;

impl AnthropicProtocol {
    /// Parse a non-streaming Anthropic response into ChatResponse.
    pub(super) fn parse_anthropic_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
        let json: Value = serde_json::from_slice(body)
            .map_err(|e| LlmError::llm(format!("Failed to parse response: {e}")))?;

        if let Some(error) = json.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            let err_type = error
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("api_error");
            return Err(LlmError::llm(format!("{err_type}: {msg}")));
        }

        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut thinking_signature = None;
        let mut tool_calls = Vec::new();

        if let Some(blocks) = json.get("content").and_then(|c| c.as_array()) {
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            content.push_str(text);
                        }
                    }
                    Some("thinking") => {
                        if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                            reasoning_content.push_str(thinking);
                        }
                        // Extract signature for multi-turn thinking
                        if let Some(sig) = block.get("signature").and_then(|s| s.as_str()) {
                            thinking_signature = Some(sig.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = block
                            .get("input")
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u
                    .get("input_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                completion_tokens: u
                    .get("output_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                total_tokens: None,
                reasoning_tokens: None,
            })
            .unwrap_or_default();

        let finish_reason = json
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "end_turn" => FinishReason::Stop,
                "tool_use" => FinishReason::ToolCalls,
                "max_tokens" => FinishReason::Length,
                "stop_sequence" => FinishReason::Stop,
                "refusal" => FinishReason::ContentFilter,
                "pause_turn" => FinishReason::Stop,
                other => FinishReason::Other(other.to_string()),
            })
            .unwrap_or(FinishReason::Stop);

        Ok(ChatResponse {
            content,
            reasoning_content: if reasoning_content.is_empty() {
                None
            } else {
                Some(reasoning_content)
            },
            thinking_signature,
            tool_calls,
            usage,
            finish_reason,
            raw: Some(json),
        })
    }
}
