//! Minimal Anthropic stream event types for SSE parsing.
//!
//! Only contains the types needed to deserialize Anthropic Messages API
//! streaming events. Request building is done with raw `serde_json::Value`.

use serde::{Deserialize, Serialize};

// ── MessagesResponse (used in MessageStart) ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MessagesResponse {
    pub usage: Usage,
    #[serde(flatten)]
    pub _rest: std::collections::HashMap<String, serde_json::Value>,
}

// ── ContentBlock (used in ContentBlockStart) ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

// ── ContentBlockDelta (used in ContentBlockDelta event) ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    /// Anthropic sends thinking signature as a separate delta
    SignatureDelta {
        signature: String,
    },
    #[serde(other)]
    Other,
}

// ── StopReason (used in MessageDelta) ──

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
}

// ── MessageDelta / MessageDeltaUsage (used in MessageDelta event) ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageDelta {
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

// ── MessagesStreamEvent (top-level SSE event) ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesStreamEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: MessageDeltaUsage,
    },
    MessageStop,
    /// DeepSeek heartbeat event (no payload)
    Ping,
    /// Anthropic error event in stream
    Error {
        error: AnthropicError,
    },
    /// Catch-all for unknown event types (future provider extensions)
    #[serde(other)]
    Unknown,
}

/// Anthropic error object in stream events
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_block_tool_use_roundtrip() {
        let block = ContentBlock::ToolUse {
            id: "tu_1".into(),
            name: "get_weather".into(),
            input: json!({"city": "Paris"}),
        };
        let serialized = serde_json::to_value(&block).unwrap();
        assert_eq!(
            serialized,
            json!({"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}})
        );
        let deserialized: ContentBlock = serde_json::from_value(serialized).unwrap();
        assert_eq!(deserialized, block);
    }

    #[test]
    fn content_block_text_catches_as_other() {
        let json = json!({"type": "text", "text": "hello"});
        let block: ContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(block, ContentBlock::Other);
    }

    #[test]
    fn content_block_delta_text() {
        let json = json!({"type": "text_delta", "text": "hello"});
        let delta: ContentBlockDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta, ContentBlockDelta::TextDelta { text: "hello".into() });
    }

    #[test]
    fn content_block_delta_thinking() {
        let json = json!({"type": "thinking_delta", "thinking": "hmm"});
        let delta: ContentBlockDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta, ContentBlockDelta::ThinkingDelta { thinking: "hmm".into() });
    }

    #[test]
    fn content_block_delta_input_json() {
        let json = json!({"type": "input_json_delta", "partial_json": "{\"q\":\""});
        let delta: ContentBlockDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta, ContentBlockDelta::InputJsonDelta { partial_json: "{\"q\":\"".into() });
    }

    #[test]
    fn content_block_delta_signature_parses_correctly() {
        let json = json!({"type": "signature_delta", "signature": "abc"});
        let delta: ContentBlockDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta, ContentBlockDelta::SignatureDelta { signature: "abc".to_string() });
    }

    #[test]
    fn stop_reason_roundtrip() {
        for (reason, expected) in [
            (StopReason::EndTurn, "end_turn"),
            (StopReason::MaxTokens, "max_tokens"),
            (StopReason::StopSequence, "stop_sequence"),
            (StopReason::ToolUse, "tool_use"),
            (StopReason::PauseTurn, "pause_turn"),
            (StopReason::Refusal, "refusal"),
        ] {
            let val = serde_json::to_value(reason).unwrap();
            assert_eq!(val, json!(expected), "serialize {reason:?}");
            let back: StopReason = serde_json::from_value(val).unwrap();
            assert_eq!(back, reason, "deserialize {reason:?}");
        }
    }

    #[test]
    fn usage_deserializes_minimal() {
        let usage: Usage = serde_json::from_value(json!({"input_tokens": 12, "output_tokens": 5})).unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert!(usage.cache_creation_input_tokens.is_none());
    }

    #[test]
    fn stream_event_message_start() {
        let json = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "stop_reason": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        });
        let event: MessagesStreamEvent = serde_json::from_value(json).unwrap();
        match event {
            MessagesStreamEvent::MessageStart { message } => {
                assert_eq!(message.usage.input_tokens, 10);
                assert_eq!(message.usage.output_tokens, 0);
            }
            other => panic!("expected MessageStart, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_content_block_start_tool_use() {
        let json = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}
        });
        let event: MessagesStreamEvent = serde_json::from_value(json).unwrap();
        match event {
            MessagesStreamEvent::ContentBlockStart { index, content_block } => {
                assert_eq!(index, 0);
                assert!(matches!(content_block, ContentBlock::ToolUse { id, .. } if id == "tu_1"));
            }
            other => panic!("expected ContentBlockStart, got {other:?}"),
        }
    }

    #[test]
    fn stream_event_message_delta_with_stop() {
        let json = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 100}
        });
        let event: MessagesStreamEvent = serde_json::from_value(json).unwrap();
        match event {
            MessagesStreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
                assert_eq!(usage.output_tokens, 100);
            }
            other => panic!("expected MessageDelta, got {other:?}"),
        }
    }

    #[test]
    fn messages_response_flatten_unknown_fields() {
        let json = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude",
            "stop_reason": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp: MessagesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp._rest["id"], "msg_1");
        assert_eq!(resp._rest["model"], "claude");
    }

    #[test]
    fn stream_event_error_parses_correctly() {
        let json = json!({
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Too many requests"
            }
        });
        let event: MessagesStreamEvent = serde_json::from_value(json).unwrap();
        match event {
            MessagesStreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
                assert_eq!(error.message, "Too many requests");
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }
}
