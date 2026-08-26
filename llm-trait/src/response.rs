//! Chat response, chat stream, and stream chunk types.

use futures_core::Stream;
use serde_json::Value;
use std::collections::BTreeMap;
use std::pin::Pin;

use super::error::LlmError;
use super::types::UsageInfo;

/// Unified non-streaming response format.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Text content
    pub content: String,

    /// Reasoning/thinking content (from StreamChunk::Thought)
    pub reasoning_content: Option<String>,

    /// Anthropic thinking signature for multi-turn conversations
    pub thinking_signature: Option<String>,

    /// Tool call list
    pub tool_calls: Vec<ToolCall>,

    /// Usage information
    pub usage: UsageInfo,

    /// Finish reason
    pub finish_reason: FinishReason,

    /// Raw response (for debugging)
    pub raw: Option<Value>,
}

/// Tool call.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Finish reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Natural completion
    Stop,
    /// Hit token limit
    Length,
    /// Requested tool calls
    ToolCalls,
    /// Content filter
    ContentFilter,
    /// Other reason
    Other(String),
}

impl FinishReason {
    pub fn from_str(s: &str) -> Self {
        match s {
            "stop" | "end_turn" => Self::Stop,
            "length" | "max_tokens" => Self::Length,
            "tool_calls" | "tool_use" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Other(s) => s,
        }
    }
}

/// Streaming response chunk.
#[derive(Clone, Debug)]
pub enum StreamChunk {
    /// Text content
    Text(String),
    /// Thinking process
    Thought(String),
    /// Anthropic thinking signature (for multi-turn)
    ThinkingSignature(String),
    /// Tool call
    ToolCall(Value),
    /// Usage information
    Usage(UsageInfo),
    /// Stream error (from API error events)
    Error(String),
    /// Stream end
    Stop {
        finish_reason: Option<String>,
    },
}

/// Streaming response wrapper.
///
/// Wraps an async Stream and provides convenient collection methods.
pub struct ChatStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>,
}

impl ChatStream {
    /// Create a new ChatStream.
    pub fn new(
        stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>,
    ) -> Self {
        Self { inner: stream }
    }

    /// Consume self and return the inner stream.
    ///
    /// Useful for adapters that need to wrap the stream in a different type.
    pub fn into_inner(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>> {
        self.inner
    }

    /// Get the next chunk.
    pub async fn next(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        use futures_util::StreamExt;
        self.inner.next().await
    }

    /// Collect all text into a string.
    ///
    /// Note: this loses tool calls and usage information.
    /// Use [`collect_response`](Self::collect_response) for the full response.
    pub async fn collect_text(mut self) -> Result<String, LlmError> {
        let mut text = String::new();
        while let Some(chunk) = self.next().await {
            match chunk? {
                StreamChunk::Text(t) => text.push_str(&t),
                StreamChunk::Stop { .. } => break,
                _ => {}
            }
        }
        Ok(text)
    }

    /// Collect full response (including tool calls, usage).
    ///
    /// Used by `GenericProvider::chat()` stream fallback mode,
    /// ensuring no tool calls or usage info is lost.
    ///
    /// Handles incremental tool call assembly: tool calls arrive across
    /// multiple chunks (first has id+name, subsequent have only argument fragments).
    pub async fn collect_response(mut self) -> Result<ChatResponse, LlmError> {
        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut thinking_signature = None;
        let mut tool_call_buf: BTreeMap<usize, ToolCall> = BTreeMap::new();
        let mut usage = UsageInfo::default();
        let mut finish_reason = FinishReason::Stop;

        while let Some(chunk) = self.next().await {
            match chunk? {
                StreamChunk::Text(t) => content.push_str(&t),
                StreamChunk::Thought(t) => reasoning_content.push_str(&t),
                StreamChunk::ThinkingSignature(sig) => {
                    thinking_signature = Some(sig);
                }
                StreamChunk::ToolCall(v) => {
                    apply_tool_call_delta(&mut tool_call_buf, &v);
                }
                StreamChunk::Usage(u) => {
                    // Accumulate: only overwrite if new value is Some
                    if u.prompt_tokens.is_some() {
                        usage.prompt_tokens = u.prompt_tokens;
                    }
                    if u.completion_tokens.is_some() {
                        usage.completion_tokens = u.completion_tokens;
                    }
                    if u.total_tokens.is_some() {
                        usage.total_tokens = u.total_tokens;
                    }
                }
                StreamChunk::Error(msg) => {
                    // Stream error from API - propagate as LlmError
                    return Err(LlmError::llm(msg));
                }
                StreamChunk::Stop {
                    finish_reason: Some(reason),
                } => {
                    finish_reason = FinishReason::from_str(&reason);
                    break;
                }
                StreamChunk::Stop { .. } => break,
            }
        }

        let tool_calls: Vec<ToolCall> = tool_call_buf.into_values().collect();

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
            raw: None,
        })
    }
}

/// Apply an incremental tool call delta to the assembly buffer.
///
/// Handles two formats:
/// - Array format (OpenAI streaming): `[{"index": 0, "id": "...", "function": {...}}]`
/// - Single object format: `{"id": "...", "name": "...", "arguments": "..."}`
///
/// Each delta may contain:
/// - `id` + `name` (or `function.name`): start of a new tool call
/// - `arguments` (or `function.arguments`) only: fragment to append to an existing tool call
fn apply_tool_call_delta(buf: &mut BTreeMap<usize, ToolCall>, value: &Value) {
    // Helper to apply a single delta item
    fn apply_one(buf: &mut BTreeMap<usize, ToolCall>, item: &Value) {
        let index = item.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

        let id = item.get("id").and_then(|i| i.as_str());

        // Support both OpenAI format (function.name) and simplified format (name directly)
        let function = item.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .or_else(|| item.get("name").and_then(|n| n.as_str()));
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(|a| a.as_str())
            .or_else(|| item.get("arguments").and_then(|a| a.as_str()));

        // Start of a new tool call
        if let (Some(id), Some(name)) = (id, name) {
            let tc = ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: arguments.unwrap_or("").to_string(),
            };
            buf.insert(index, tc);
            return;
        }

        // Argument fragment for an existing tool call
        if let Some(args_fragment) = arguments {
            if let Some(tc) = buf.get_mut(&index) {
                tc.arguments.push_str(args_fragment);
            }
        }
    }

    // Unwrap {"delta": {"tool_calls": [...]}} wrapper if present.
    // Both the llm-engine consumer and the Anthropic provider use this format.
    let unwrapped = value
        .get("delta")
        .and_then(|d| d.get("tool_calls"))
        .unwrap_or(value);

    if let Some(arr) = unwrapped.as_array() {
        for item in arr {
            apply_one(buf, item);
        }
    } else {
        apply_one(buf, unwrapped);
    }
}

/// Extract tool call information from a Value.
///
/// Supports two formats:
/// - OpenAI format: `[{"id": "...", "function": {"name": "...", "arguments": "..."}}]`
/// - Simplified format: `{"id": "...", "name": "...", "arguments": "..."}`
pub fn extract_tool_calls(value: &Value) -> Option<Vec<ToolCall>> {
    // OpenAI format: array
    if let Some(arr) = value.as_array() {
        let calls: Vec<ToolCall> = arr
            .iter()
            .filter_map(|item| {
                let id = item.get("id")?.as_str()?.to_string();
                let function = item.get("function")?;
                let name = function.get("name")?.as_str()?.to_string();
                let arguments = function
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();
        return if calls.is_empty() { None } else { Some(calls) };
    }

    // Simplified format: single object
    let id = value.get("id")?.as_str()?.to_string();
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .to_string();
    Some(vec![ToolCall {
        id,
        name,
        arguments,
    }])
}

/// Parse finish reason string to FinishReason enum.
///
/// Convenience wrapper around [`FinishReason::from_str`].
pub fn parse_finish_reason(s: &str) -> FinishReason {
    FinishReason::from_str(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FinishReason ──

    #[test]
    fn finish_reason_from_str() {
        assert_eq!(FinishReason::from_str("stop"), FinishReason::Stop);
        assert_eq!(FinishReason::from_str("end_turn"), FinishReason::Stop);
        assert_eq!(FinishReason::from_str("length"), FinishReason::Length);
        assert_eq!(FinishReason::from_str("max_tokens"), FinishReason::Length);
        assert_eq!(FinishReason::from_str("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(FinishReason::from_str("tool_use"), FinishReason::ToolCalls);
        assert_eq!(
            FinishReason::from_str("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(
            FinishReason::from_str("unknown_reason"),
            FinishReason::Other("unknown_reason".to_string())
        );
    }

    #[test]
    fn finish_reason_as_str() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
        assert_eq!(FinishReason::ToolCalls.as_str(), "tool_calls");
        assert_eq!(FinishReason::ContentFilter.as_str(), "content_filter");
    }

    // ── ChatStream::collect_text ──

    #[tokio::test]
    async fn collect_text_basic() {
        let chunks = vec![
            Ok(StreamChunk::Text("hello ".into())),
            Ok(StreamChunk::Text("world".into())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn collect_text_ignores_non_text() {
        let chunks = vec![
            Ok(StreamChunk::Thought("thinking...".into())),
            Ok(StreamChunk::Text("visible".into())),
            Ok(StreamChunk::ToolCall(serde_json::json!({"name": "shell"}))),
            Ok(StreamChunk::Usage(UsageInfo {
                prompt_tokens: Some(10),
                ..Default::default()
            })),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "visible");
    }

    #[tokio::test]
    async fn collect_text_stops_on_stop_chunk() {
        let chunks = vec![
            Ok(StreamChunk::Text("before".into())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
            Ok(StreamChunk::Text("after".into())),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "before");
    }

    #[tokio::test]
    async fn collect_text_empty_stream() {
        let chunks: Vec<Result<StreamChunk, LlmError>> = vec![];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let text = stream.collect_text().await.unwrap();
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn collect_text_error_propagates() {
        let chunks: Vec<Result<StreamChunk, LlmError>> = vec![
            Ok(StreamChunk::Text("before".into())),
            Err(LlmError::llm("stream broke")),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let result = stream.collect_text().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stream broke"));
    }

    // ── ChatStream::collect_response ──

    #[tokio::test]
    async fn collect_response_full_lifecycle() {
        let chunks = vec![
            Ok(StreamChunk::Text("Hello ".into())),
            Ok(StreamChunk::Text("world!".into())),
            Ok(StreamChunk::ToolCall(serde_json::json!([
                {
                    "id": "call_1",
                    "function": {
                        "name": "shell",
                        "arguments": "{\"cmd\": \"ls\"}"
                    }
                }
            ]))),
            Ok(StreamChunk::Usage(UsageInfo {
                prompt_tokens: Some(100),
                completion_tokens: Some(50),
                total_tokens: Some(150),
            })),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.content, "Hello world!");
        assert!(response.reasoning_content.is_none());
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "shell");
        assert_eq!(response.tool_calls[0].arguments, "{\"cmd\": \"ls\"}");
        assert_eq!(response.usage.prompt_tokens, Some(100));
        assert_eq!(response.usage.completion_tokens, Some(50));
        assert_eq!(response.usage.total_tokens, Some(150));
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn collect_response_with_thought() {
        let chunks = vec![
            Ok(StreamChunk::Thought("thinking...".into())),
            Ok(StreamChunk::Text("answer".into())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.content, "answer");
        assert_eq!(response.reasoning_content.as_deref(), Some("thinking..."));
    }

    #[tokio::test]
    async fn collect_response_usage_accumulate() {
        // Simulate Anthropic: message_start has prompt_tokens, message_delta has completion_tokens
        let chunks = vec![
            Ok(StreamChunk::Usage(UsageInfo {
                prompt_tokens: Some(20),
                completion_tokens: Some(0),
                total_tokens: None,
            })),
            Ok(StreamChunk::Text("ok".into())),
            Ok(StreamChunk::Usage(UsageInfo {
                prompt_tokens: None,
                completion_tokens: Some(7),
                total_tokens: None,
            })),
            Ok(StreamChunk::Stop {
                finish_reason: Some("end_turn".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        // prompt_tokens should be preserved from first Usage chunk
        assert_eq!(response.usage.prompt_tokens, Some(20));
        assert_eq!(response.usage.completion_tokens, Some(7));
    }

    #[tokio::test]
    async fn collect_response_text_only() {
        let chunks = vec![
            Ok(StreamChunk::Text("Just text".into())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.content, "Just text");
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn collect_response_no_stop_chunk() {
        let chunks: Vec<Result<StreamChunk, LlmError>> = vec![
            Ok(StreamChunk::Text("text".into())),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.content, "text");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn collect_response_tool_calls_format() {
        let chunks = vec![
            Ok(StreamChunk::ToolCall(serde_json::json!({
                "id": "call_abc",
                "name": "read_file",
                "arguments": "{\"path\": \"/tmp/test.txt\"}"
            }))),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let response = stream.collect_response().await.unwrap();

        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_abc");
        assert_eq!(response.tool_calls[0].name, "read_file");
    }

    #[tokio::test]
    async fn collect_response_error_propagates() {
        let chunks: Vec<Result<StreamChunk, LlmError>> = vec![
            Ok(StreamChunk::Text("before".into())),
            Err(LlmError::llm("broken")),
        ];
        let stream = ChatStream::new(Box::pin(futures_util::stream::iter(chunks)));
        let result = stream.collect_response().await;
        assert!(result.is_err());
    }

    // ── extract_tool_calls ──

    #[test]
    fn extract_tool_calls_openai_format() {
        let value = serde_json::json!([
            {
                "id": "call_1",
                "function": {
                    "name": "shell",
                    "arguments": "{\"cmd\": \"ls\"}"
                }
            },
            {
                "id": "call_2",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"/tmp\"}"
                }
            }
        ]);
        let calls = extract_tool_calls(&value).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].id, "call_2");
        assert_eq!(calls[1].name, "read_file");
    }

    #[test]
    fn extract_tool_calls_simple_format() {
        let value = serde_json::json!({
            "id": "call_1",
            "name": "shell",
            "arguments": "{\"cmd\": \"ls\"}"
        });
        let calls = extract_tool_calls(&value).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn extract_tool_calls_invalid_returns_none() {
        let value = serde_json::json!("just a string");
        assert!(extract_tool_calls(&value).is_none());
    }

    #[test]
    fn extract_tool_calls_empty_array_returns_none() {
        let value = serde_json::json!([]);
        assert!(extract_tool_calls(&value).is_none());
    }

    // ── proptest: parse_finish_reason ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn from_str_never_panics(s in ".*") {
                let _ = FinishReason::from_str(&s);
            }

            #[test]
            fn from_str_known_values(
                variant in proptest::sample::select(vec![
                    ("stop", "Stop"),
                    ("end_turn", "Stop"),
                    ("length", "Length"),
                    ("max_tokens", "Length"),
                    ("tool_calls", "ToolCalls"),
                    ("tool_use", "ToolCalls"),
                    ("content_filter", "ContentFilter"),
                ])
            ) {
                let (s, expected_name) = variant;
                let fr = FinishReason::from_str(s);
                match (expected_name, &fr) {
                    ("Stop", FinishReason::Stop) => {}
                    ("Length", FinishReason::Length) => {}
                    ("ToolCalls", FinishReason::ToolCalls) => {}
                    ("ContentFilter", FinishReason::ContentFilter) => {}
                    _ => panic!("from_str({:?}) = {:?}, expected {}", s, fr, expected_name),
                }
            }

            #[test]
            fn from_str_unknown_returns_other(s in "[a-z_]{1,30}") {
                // Filter out known strings
                if ["stop", "end_turn", "length", "max_tokens", "tool_calls", "tool_use", "content_filter"].contains(&s.as_str()) {
                    return Ok(());
                }
                let fr = FinishReason::from_str(&s);
                match fr {
                    FinishReason::Other(inner) => assert_eq!(inner, s),
                    other => panic!("expected Other({:?}), got {:?}", s, other),
                }
            }
        }
    }
}
