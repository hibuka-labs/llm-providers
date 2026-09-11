//! Chat request types.

use serde_json::Value;

use super::message::ChatMessage;
use super::reasoning::ReasoningConfig;

/// Response format configuration.
#[derive(Clone, Debug)]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema { name: String, schema: Value },
}

impl ResponseFormat {
    pub fn to_api_value(&self) -> Value {
        match self {
            ResponseFormat::JsonObject => {
                serde_json::json!({ "type": "json_object" })
            }
            ResponseFormat::JsonSchema { name, schema } => {
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                    }
                })
            }
        }
    }
}

/// Unified chat request format.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Message list (supports multimodal content)
    pub messages: Vec<ChatMessage>,

    /// Available tools
    pub tools: Vec<Value>,

    /// Reasoning/thinking configuration
    pub reasoning: Option<ReasoningConfig>,

    /// Response format configuration
    pub response_format: Option<ResponseFormat>,

    /// Optional model override.
    ///
    /// - `None`: Use the provider's default model
    /// - `Some(name)`: Use the specified model for this request
    ///
    /// This enables multi-model routing: the application can set different
    /// models for different scenes (e.g., "lite" for sub-agents, "advanced"
    /// for architecture decisions) without creating multiple providers.
    pub model: Option<String>,
}

impl ChatRequest {
    /// Create a simple text request.
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            reasoning: None,
            response_format: None,
            model: None,
        }
    }

    /// Set model override.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add tools.
    pub fn with_tools(mut self, tools: Vec<Value>) -> Self {
        self.tools = tools;
        self
    }

    /// Add reasoning configuration.
    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set response format.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }
}

#[cfg(test)]
mod response_format_tests {
    use super::*;

    #[test]
    fn json_object_to_api_value() {
        assert_eq!(
            ResponseFormat::JsonObject.to_api_value(),
            serde_json::json!({ "type": "json_object" })
        );
    }

    #[test]
    fn json_schema_to_api_value() {
        let schema = serde_json::json!({"type": "object"});
        let value = ResponseFormat::JsonSchema {
            name: "answer".to_string(),
            schema: schema.clone(),
        }
        .to_api_value();

        assert_eq!(value["type"], "json_schema");
        assert_eq!(value["json_schema"]["name"], "answer");
        assert_eq!(value["json_schema"]["schema"], schema);
    }

    #[test]
    fn with_response_format_stores_the_format() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_response_format(ResponseFormat::JsonObject);
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonObject)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_new() {
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]);
        assert_eq!(req.messages.len(), 1);
        assert!(req.tools.is_empty());
        assert!(req.reasoning.is_none());
        assert!(req.response_format.is_none());
        assert!(req.model.is_none());
    }

    #[test]
    fn chat_request_with_model() {
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_model("mimo-v2.5-pro");
        assert_eq!(req.model.as_deref(), Some("mimo-v2.5-pro"));
    }

    #[test]
    fn chat_request_with_tools() {
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "test"}})];
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(tools.clone());
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0], tools[0]);
    }

    #[test]
    fn chat_request_with_reasoning() {
        let reasoning = ReasoningConfig {
            enabled: Some(true),
            budget_tokens: Some(4096),
            effort: None,
        };
        let req = ChatRequest::new(vec![ChatMessage::user("hello")]).with_reasoning(reasoning);
        assert!(req.reasoning.is_some());
        assert_eq!(req.reasoning.as_ref().unwrap().budget_tokens, Some(4096));
    }

    #[test]
    fn chat_request_builder_chain() {
        let tools = vec![serde_json::json!({"name": "tool1"})];
        let reasoning = ReasoningConfig {
            enabled: Some(true),
            ..Default::default()
        };
        let req = ChatRequest::new(vec![ChatMessage::user("hello")])
            .with_tools(tools)
            .with_reasoning(reasoning);
        assert_eq!(req.tools.len(), 1);
        assert!(req.reasoning.is_some());
    }
}
