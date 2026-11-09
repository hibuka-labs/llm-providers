use llm_trait::{LlmError, StreamChunk, UsageInfo};

use crate::protocol::anthropic::types::{
    ContentBlock, ContentBlockDelta, MessagesStreamEvent, StopReason,
};

use super::AnthropicProtocol;

impl AnthropicProtocol {
    /// Convert an Anthropic stream event to StreamChunk(s).
    /// Returns Err for error events (propagated as stream-level errors).
    pub(super) fn convert_event(event: MessagesStreamEvent) -> Result<Vec<StreamChunk>, LlmError> {
        let chunks = match event {
            MessagesStreamEvent::MessageStart { message } => {
                vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: Some(message.usage.input_tokens),
                    completion_tokens: Some(message.usage.output_tokens),
                    total_tokens: None,
                    reasoning_tokens: None,
                })]
            }
            MessagesStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ContentBlock::ToolUse { id, name, .. } => {
                    vec![StreamChunk::ToolCall(serde_json::json!({
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "function": {
                                    "name": name,
                                    "arguments": "",
                                }
                            }]
                        }
                    }))]
                }
                _ => vec![],
            },
            MessagesStreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentBlockDelta::TextDelta { text } => {
                    vec![StreamChunk::Text(text)]
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    vec![StreamChunk::ToolCall(serde_json::json!({
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {
                                    "arguments": partial_json,
                                }
                            }]
                        }
                    }))]
                }
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    vec![StreamChunk::Thought(thinking)]
                }
                ContentBlockDelta::SignatureDelta { signature } => {
                    vec![StreamChunk::ThinkingSignature(signature)]
                }
                _ => vec![],
            },
            MessagesStreamEvent::ContentBlockStop { .. } => vec![],
            MessagesStreamEvent::MessageDelta { delta, usage } => {
                // `usage.input_tokens` is optional per protocol but some
                // compatible endpoints (DashScope) carry the full-context
                // count here; the aggregator merges it over the
                // message_start value.
                let mut chunks = vec![StreamChunk::Usage(UsageInfo {
                    prompt_tokens: usage.input_tokens,
                    completion_tokens: Some(usage.output_tokens),
                    total_tokens: None,
                    reasoning_tokens: None,
                })];

                if let Some(reason) = delta.stop_reason {
                    let finish_reason = match reason {
                        StopReason::EndTurn => "end_turn",
                        StopReason::ToolUse => "tool_use",
                        StopReason::MaxTokens => "max_tokens",
                        StopReason::StopSequence => "stop_sequence",
                        StopReason::PauseTurn => "pause_turn",
                        StopReason::Refusal => "refusal",
                    };
                    if reason == StopReason::MaxTokens {
                        tracing::warn!(
                            finish_reason,
                            "stream truncated by token limit — tool call arguments may be incomplete"
                        );
                    }
                    chunks.push(StreamChunk::Stop {
                        finish_reason: Some(finish_reason.to_string()),
                    });
                }

                chunks
            }
            MessagesStreamEvent::MessageStop => vec![],
            MessagesStreamEvent::Ping => vec![],
            MessagesStreamEvent::Error { error } => {
                tracing::error!(
                    error_type = %error.error_type,
                    message = %error.message,
                    "Anthropic stream error event"
                );
                return Err(LlmError::llm(format!(
                    "{}: {}",
                    error.error_type, error.message
                )));
            }
            MessagesStreamEvent::Unknown => {
                tracing::debug!(
                    "Ignoring unknown Anthropic-protocol SSE event (provider extensions like DeepSeek ping)"
                );
                vec![]
            }
        };
        Ok(chunks)
    }
}
