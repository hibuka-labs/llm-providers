use serde_json::Value;

use llm_trait::StreamChunk;

use super::OpenAiProtocol;

impl OpenAiProtocol {
    /// Convert a single OpenAI SSE delta to StreamChunk(s).
    pub(super) fn convert_delta(delta: &Value, index: usize) -> Vec<StreamChunk> {
        let mut chunks = Vec::new();

        if let Some(content) = delta.get("content").and_then(|c| c.as_str())
            && !content.is_empty()
        {
            chunks.push(StreamChunk::Text(content.to_string()));
        }

        if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str())
            && !reasoning.is_empty()
        {
            chunks.push(StreamChunk::Thought(reasoning.to_string()));
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let tc_index = tc
                    .get("index")
                    .and_then(|i| i.as_u64())
                    .unwrap_or(index as u64) as usize;
                let mut delta_obj = serde_json::json!({});

                if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                    delta_obj["id"] = Value::String(id.to_string());
                }

                if let Some(function) = tc.get("function") {
                    let mut func_delta = serde_json::json!({});
                    if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                        func_delta["name"] = Value::String(name.to_string());
                    }
                    if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                        func_delta["arguments"] = Value::String(args.to_string());
                    }
                    delta_obj["function"] = func_delta;
                }

                chunks.push(StreamChunk::ToolCall(serde_json::json!({
                    "delta": {
                        "tool_calls": [{
                            "index": tc_index,
                            "id": delta_obj.get("id"),
                            "function": delta_obj.get("function"),
                        }]
                    }
                })));
            }
        }

        chunks
    }
}
