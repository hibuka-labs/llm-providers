use serde_json::Value;

use llm_trait::{ChatMessage, ChatRequest, ImageAttachment, ImageDetail, LlmError, ReasoningSpec};

use super::OpenAiProtocol;

impl OpenAiProtocol {
    /// Build OpenAI Chat Completions request body from ChatRequest.
    pub(super) fn build_openai_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<Value, LlmError> {
        let messages = Self::convert_messages(&request.messages);
        let tools = Self::convert_tools(&request.tools);

        tracing::debug!(
            model = %self.model,
            msg_count = messages.len(),
            tool_count = tools.len(),
            has_reasoning = request.reasoning.is_some(),
            stream = stream,
            "building OpenAI request"
        );

        // Priority: user config > profile > default
        // If user explicitly set max_tokens (non-default), user config wins
        // Otherwise use profile value, fallback to default
        let default_max_tokens = 16_384;
        let effective_max_tokens = if self.max_tokens != default_max_tokens {
            // User explicitly set max_tokens, user config wins
            self.max_tokens
        } else {
            // User didn't set explicitly, use profile value or default
            self.profile
                .as_ref()
                .and_then(|p| p.capabilities.max_output_tokens)
                .unwrap_or(self.max_tokens)
        };

        tracing::debug!(
            model = %self.model,
            effective_max_tokens,
            configured_max_tokens = self.max_tokens,
            has_profile = self.profile.is_some(),
            "OpenAI request max_tokens resolved"
        );

        // Use request.model if specified, otherwise use default model
        let effective_model = request.model.as_deref().unwrap_or(&self.model);

        let mut body = serde_json::json!({
            "model": effective_model,
            "messages": messages,
            "stream": stream,
            "max_tokens": effective_max_tokens,
        });

        // Request usage data in stream mode
        if stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| LlmError::llm(format!("Failed to serialize tools: {e}")))?;
        }

        // Handle reasoning: only send if profile says this model supports Effort mode
        if let Some(ref rc) = request.reasoning
            && let Some(ref profile) = self.profile
        {
            let spec = rc.to_spec(profile.reasoning_mode);
            tracing::debug!(
                model = %self.model,
                reasoning_mode = ?profile.reasoning_mode,
                spec = ?spec,
                "reasoning config resolved"
            );
            if let ReasoningSpec::Effort(effort) = spec {
                let effort_str = match effort {
                    llm_trait::ReasoningEffort::Low => "low",
                    llm_trait::ReasoningEffort::Medium => "medium",
                    llm_trait::ReasoningEffort::High => "high",
                    // ReasoningEffort::None is filtered out by to_spec(),
                    // so ReasoningSpec::Effort(None) never reaches here.
                    llm_trait::ReasoningEffort::None => unreachable!(),
                    llm_trait::ReasoningEffort::XHigh => "high",
                };
                body["reasoning_effort"] = Value::String(effort_str.to_string());
            }
        }

        Ok(body)
    }

    /// Convert ChatMessages to OpenAI message format.
    pub(super) fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => {
                    result.push(serde_json::json!({
                        "role": "system",
                        "content": content
                    }));
                }
                ChatMessage::User {
                    content, images, ..
                } => {
                    if images.is_empty() {
                        result.push(serde_json::json!({
                            "role": "user",
                            "content": content
                        }));
                    } else {
                        let mut parts = vec![serde_json::json!({
                            "type": "text",
                            "text": content
                        })];

                        for img in images {
                            match img {
                                ImageAttachment::Url { url, detail } => {
                                    let mut image_part = serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": url }
                                    });
                                    if let Some(d) = detail {
                                        let detail_str = match d {
                                            ImageDetail::Low => "low",
                                            ImageDetail::High => "high",
                                            ImageDetail::Auto => "auto",
                                        };
                                        image_part["image_url"]["detail"] =
                                            Value::String(detail_str.to_string());
                                    }
                                    parts.push(image_part);
                                }
                                ImageAttachment::Base64 {
                                    data,
                                    media_type,
                                    detail,
                                } => {
                                    let media_type = media_type.as_deref().unwrap_or("image/jpeg");
                                    let url = format!("data:{media_type};base64,{data}");
                                    let mut image_part = serde_json::json!({
                                        "type": "image_url",
                                        "image_url": { "url": url }
                                    });
                                    if let Some(d) = detail {
                                        let detail_str = match d {
                                            ImageDetail::Low => "low",
                                            ImageDetail::High => "high",
                                            ImageDetail::Auto => "auto",
                                        };
                                        image_part["image_url"]["detail"] =
                                            Value::String(detail_str.to_string());
                                    }
                                    parts.push(image_part);
                                }
                            }
                        }

                        result.push(serde_json::json!({
                            "role": "user",
                            "content": parts
                        }));
                    }
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    let mut msg = serde_json::json!({
                        "role": "assistant"
                    });

                    if let Some(text) = content.as_ref().filter(|t| !t.is_empty()) {
                        msg["content"] = Value::String(text.clone());
                    }

                    if let Some(tc) = tool_calls {
                        let openai_tool_calls: Vec<Value> = tc
                            .iter()
                            .map(|t| {
                                serde_json::json!({
                                    "id": t.id,
                                    "type": "function",
                                    "function": {
                                        "name": t.name,
                                        "arguments": t.arguments
                                    }
                                })
                            })
                            .collect();
                        msg["tool_calls"] = Value::Array(openai_tool_calls);
                    }

                    result.push(msg);
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                    images,
                    ..
                } => {
                    // A `role:"tool"` message is string-only here: the shape
                    // cannot carry a picture. Say so rather than drop it —
                    // a model that knows it missed a screenshot can ask for
                    // it again; one that never heard of it is silently blind.
                    let body = if images.is_empty() {
                        content.clone()
                    } else {
                        format!(
                            "{content}\n[note: this result also carried {} image(s), \
                             which this provider's tool-result shape cannot deliver]",
                            images.len()
                        )
                    };
                    result.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": body
                    }));
                }
                ChatMessage::Custom { .. } => {
                    // Custom messages are metadata-only, not forwarded to LLM
                }
            }
        }

        // Merge consecutive same-role messages
        let mut merged: Vec<Value> = Vec::with_capacity(result.len());
        for msg in result {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            let can_merge = if let Some(prev) = merged.last() {
                let prev_role = prev.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role == prev_role && role != "tool"
            } else {
                false
            };

            if can_merge && let Some(prev) = merged.last_mut() {
                if role == "assistant" {
                    let prev_has_tc = prev.get("tool_calls").is_some_and(|t| t.is_array());
                    let new_has_tc = msg.get("tool_calls").is_some_and(|t| t.is_array());

                    // Don't merge tool_calls+text into one message.
                    // OpenAI API requires tool_calls messages to have null/empty content.
                    if prev_has_tc != new_has_tc {
                        // One has tool_calls, the other has text — don't merge
                        merged.push(msg);
                        continue;
                    }

                    let prev_content = prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let new_content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let combined_content = if prev_content.is_empty() {
                        new_content.to_string()
                    } else if new_content.is_empty() {
                        prev_content.to_string()
                    } else {
                        format!("{}\n{}", prev_content, new_content)
                    };
                    if !combined_content.is_empty() {
                        prev["content"] = Value::String(combined_content);
                    }

                    if let Some(new_tc) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        if let Some(prev_tc) =
                            prev.get_mut("tool_calls").and_then(|t| t.as_array_mut())
                        {
                            prev_tc.extend(new_tc.iter().cloned());
                        } else {
                            prev["tool_calls"] = Value::Array(new_tc.clone());
                        }
                    }
                    continue;
                }
                if role == "user" {
                    let prev_is_array = prev.get("content").is_some_and(|c| c.is_array());
                    let new_is_array = msg.get("content").is_some_and(|c| c.is_array());

                    if prev_is_array || new_is_array {
                        // Either side has array content (images) — merge as arrays
                        let mut prev_parts: Vec<Value> = prev
                            .get("content")
                            .and_then(|c| c.as_array())
                            .cloned()
                            .unwrap_or_else(|| {
                                // Convert string content to a text part
                                prev.get("content")
                                    .and_then(|c| c.as_str())
                                    .filter(|s| !s.is_empty())
                                    .map(|s| vec![serde_json::json!({"type": "text", "text": s})])
                                    .unwrap_or_default()
                            });
                        let new_parts: Vec<Value> = msg
                            .get("content")
                            .and_then(|c| c.as_array())
                            .cloned()
                            .unwrap_or_else(|| {
                                msg.get("content")
                                    .and_then(|c| c.as_str())
                                    .filter(|s| !s.is_empty())
                                    .map(|s| vec![serde_json::json!({"type": "text", "text": s})])
                                    .unwrap_or_default()
                            });
                        prev_parts.extend(new_parts);
                        prev["content"] = Value::Array(prev_parts);
                    } else {
                        // Both are strings — merge as strings
                        let prev_content =
                            prev.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let new_content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let combined = if prev_content.is_empty() {
                            new_content.to_string()
                        } else if new_content.is_empty() {
                            prev_content.to_string()
                        } else {
                            format!("{}\n{}", prev_content, new_content)
                        };
                        prev["content"] = Value::String(combined);
                    }
                    continue;
                }
            }

            merged.push(msg);
        }

        merged
    }

    /// Convert tools to OpenAI format.
    pub(super) fn convert_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .map(|tool| {
                if tool.get("type").is_some() && tool.get("function").is_some() {
                    return tool.clone();
                }
                serde_json::json!({
                    "type": "function",
                    "function": tool
                })
            })
            .collect()
    }
}
