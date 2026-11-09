use serde_json::Value;

use llm_trait::{ChatMessage, ChatRequest, ImageAttachment, LlmError, ReasoningSpec};

use super::AnthropicProtocol;

/// One attachment → one Anthropic `image` content block. The same shape
/// wherever a picture rides a message: a user turn or a tool_result.
fn image_block(img: &ImageAttachment) -> Value {
    match img {
        ImageAttachment::Url { url, .. } => serde_json::json!({
            "type": "image",
            "source": { "type": "url", "url": url }
        }),
        ImageAttachment::Base64 {
            data, media_type, ..
        } => {
            let media_type = media_type.as_deref().unwrap_or("image/jpeg");
            serde_json::json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data }
            })
        }
    }
}

impl AnthropicProtocol {
    /// Build Anthropic MessagesRequest from ChatRequest.
    pub(super) fn build_anthropic_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<Value, LlmError> {
        let (system, messages) = Self::convert_messages(&request.messages);
        let tools = Self::convert_tools(&request.tools);

        tracing::debug!(
            model = %self.model,
            msg_count = messages.len(),
            tool_count = tools.len(),
            has_system = system.is_some(),
            has_reasoning = request.reasoning.is_some(),
            stream = stream,
            "building Anthropic request"
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

        // Use request.model if specified, otherwise use default model
        let effective_model = request.model.as_deref().unwrap_or(&self.model);

        let mut body = serde_json::json!({
            "model": effective_model,
            "max_tokens": effective_max_tokens,
            "messages": messages,
            "stream": stream,
        });

        if let Some(sys) = system {
            body["system"] = Value::String(sys);
        }

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(&tools)
                .map_err(|e| LlmError::llm(format!("Failed to serialize tools: {e}")))?;
            body["tool_choice"] = serde_json::json!({"type": "auto"});
        }

        // Handle reasoning: only send if profile says this model supports Thinking mode
        if let Some(ref rc) = request.reasoning
            && let Some(ref profile) = self.profile
        {
            let spec = rc.to_spec(profile.reasoning_mode);
            tracing::debug!(
                model = %self.model,
                reasoning_mode = ?profile.reasoning_mode,
                spec = ?spec,
                "Anthropic reasoning config resolved"
            );
            if let ReasoningSpec::Thinking { budget_tokens } = spec {
                let budget = budget_tokens as u32;
                body["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                });
                if budget >= effective_max_tokens {
                    tracing::warn!(
                        budget_tokens = budget,
                        original_max_tokens = self.max_tokens,
                        "budget_tokens >= max_tokens, auto-increasing max_tokens to budget+1"
                    );
                    body["max_tokens"] = Value::Number((budget + 1).into());
                }
            }
        }

        Ok(body)
    }

    /// Convert ChatMessages to Anthropic message format.
    pub(super) fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
        let mut system_prompt: Option<String> = None;
        let mut result: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                ChatMessage::System { content, .. } => match &mut system_prompt {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(content);
                    }
                    None => {
                        system_prompt = Some(content.clone());
                    }
                },
                ChatMessage::User {
                    content, images, ..
                } => {
                    let mut blocks = vec![serde_json::json!({
                        "type": "text",
                        "text": content
                    })];

                    for img in images {
                        blocks.push(image_block(img));
                    }

                    result.push(serde_json::json!({
                        "role": "user",
                        "content": blocks
                    }));
                }
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    thinking_signature,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();

                    // Anthropic requires thinking blocks with signature to be sent back
                    // in multi-turn conversations
                    if let (Some(thinking), Some(sig)) = (&reasoning_content, &thinking_signature) {
                        blocks.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": sig
                        }));
                    }

                    if let Some(text) = content.as_ref().filter(|t| !t.is_empty()) {
                        blocks.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }

                    if let Some(tc) = tool_calls {
                        for t in tc {
                            let input: Value = serde_json::from_str(&t.arguments)
                                .unwrap_or(Value::Object(Default::default()));
                            blocks.push(serde_json::json!({
                                "type": "tool_use",
                                "id": t.id,
                                "name": t.name,
                                "input": input
                            }));
                        }
                    }

                    if !blocks.is_empty() {
                        result.push(serde_json::json!({
                            "role": "assistant",
                            "content": blocks
                        }));
                    }
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                    images,
                    ..
                } => {
                    // A tool_result's content may be a block list, so the
                    // pictures a tool handed back travel beside its words —
                    // the model looks at them in the same turn as the result.
                    let body = if images.is_empty() {
                        serde_json::Value::String(content.clone())
                    } else {
                        let mut blocks = vec![serde_json::json!({
                            "type": "text",
                            "text": content
                        })];
                        for img in images {
                            blocks.push(image_block(img));
                        }
                        serde_json::Value::Array(blocks)
                    };
                    result.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": body
                        }]
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
                // Merge consecutive user or assistant messages
                (role == "user" && prev_role == "user")
                    || (role == "assistant" && prev_role == "assistant")
            } else {
                false
            };

            if can_merge
                && let Some(prev) = merged.last_mut()
                && let (Some(prev_content), Some(new_content)) = (
                    prev.get_mut("content").and_then(|c| c.as_array_mut()),
                    msg.get("content").and_then(|c| c.as_array()),
                )
            {
                prev_content.extend(new_content.iter().cloned());
                continue;
            }

            merged.push(msg);
        }

        (system_prompt, merged)
    }

    /// Convert OpenAI-style tools to Anthropic tool format.
    pub(super) fn convert_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .filter_map(|tool| {
                let func = tool.get("function")?;
                let name = func.get("name")?.as_str()?;
                let description = func
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let input_schema = func
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object"}));
                Some(serde_json::json!({
                    "name": name,
                    "description": description,
                    "input_schema": input_schema
                }))
            })
            .collect()
    }
}
