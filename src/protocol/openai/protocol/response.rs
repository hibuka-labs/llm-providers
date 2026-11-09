use serde_json::Value;

use llm_trait::{ChatResponse, FinishReason, LlmError, ToolCall, UsageInfo};

use super::OpenAiProtocol;

impl OpenAiProtocol {
    /// Parse a non-streaming OpenAI response into ChatResponse.
    pub(super) fn parse_openai_response(body: &[u8]) -> Result<ChatResponse, LlmError> {
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

        let choice = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| LlmError::llm("No choices in response"))?;

        let message = choice
            .get("message")
            .ok_or_else(|| LlmError::llm("No message in choice"))?;

        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Extract reasoning_content (for o1/o3-style and DeepSeek models)
        let reasoning_content = message
            .get("reasoning_content")
            .and_then(|r| r.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let tool_calls = message
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let id = tc.get("id")?.as_str()?.to_string();
                        let function = tc.get("function")?;
                        let name = function.get("name")?.as_str()?.to_string();
                        let arguments = function
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        Some(ToolCall {
                            id,
                            name,
                            arguments,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let usage = json
            .get("usage")
            .map(|u| UsageInfo {
                prompt_tokens: u
                    .get("prompt_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                completion_tokens: u
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                total_tokens: u
                    .get("total_tokens")
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
                reasoning_tokens: u
                    .get("completion_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|t| t.as_u64())
                    .map(|n| n as u32),
            })
            .unwrap_or_default();

        let finish_reason = choice
            .get("finish_reason")
            .and_then(|r| r.as_str())
            .map(|r| match r {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "tool_calls" => FinishReason::ToolCalls,
                "content_filter" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            })
            .unwrap_or(FinishReason::Stop);

        Ok(ChatResponse {
            content,
            reasoning_content,
            thinking_signature: None, // OpenAI doesn't use thinking signatures
            tool_calls,
            usage,
            finish_reason,
            raw: Some(json),
        })
    }
}
