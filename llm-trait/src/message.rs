//! Message types for LLM conversations.
//!
//! These types define the conversation format shared across all LLM providers.

use serde::{Deserialize, Serialize};

/// A tool call embedded in an assistant message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallMessage {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Image attachment for multimodal messages.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImageAttachment {
    Url {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    Base64 {
        data: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

/// Image detail level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

/// A chat message in a conversation.
///
/// Supports system, user, assistant, tool, and custom message types.
/// Each variant carries the data needed for LLM API calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChatMessage {
    System {
        content: String,
        /// Ephemeral messages are cleaned up after each turn and skipped during persistence.
        #[serde(default, skip_serializing)]
        ephemeral: bool,
    },
    User {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageAttachment>,
        /// Ephemeral messages are cleaned up after each turn and skipped during persistence.
        #[serde(default, skip_serializing)]
        ephemeral: bool,
    },
    Assistant {
        content: Option<String>,
        reasoning_content: Option<String>,
        /// Anthropic requires thinking blocks with signature to be sent back
        /// in multi-turn conversations. This field stores the signature.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        tool_calls: Option<Vec<ToolCallMessage>>,
    },
    Tool {
        tool_call_id: String,
        name: Option<String>,
        content: String,
    },
    /// Application-defined message type for extensibility.
    ///
    /// Custom messages are preserved in the transcript but filtered out
    /// by the default conversion before being sent to the LLM provider.
    Custom {
        role: String,
        data: serde_json::Value,
    },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            ephemeral: false,
        }
    }

    /// Create an ephemeral system message: auto-cleaned after turn, not persisted.
    pub fn system_ephemeral(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            ephemeral: true,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
            ephemeral: false,
        }
    }

    /// Create an ephemeral user message: auto-cleaned after turn, not persisted.
    pub fn user_ephemeral(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
            ephemeral: true,
        }
    }

    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self::User {
            content: content.into(),
            images,
            ephemeral: false,
        }
    }

    /// Whether this is an ephemeral message (auto-cleaned after turn, not persisted).
    pub fn is_ephemeral(&self) -> bool {
        match self {
            Self::System { ephemeral, .. } => *ephemeral,
            Self::User { ephemeral, .. } => *ephemeral,
            _ => false,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        }
    }

    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning: impl Into<String>,
    ) -> Self {
        Self::Assistant {
            content: Some(content.into()),
            reasoning_content: Some(reasoning.into()),
            thinking_signature: None,
            tool_calls: None,
        }
    }

    pub fn assistant_tool_call(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::Assistant {
            content: None,
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: Some(vec![ToolCallMessage {
                id: tool_call_id.into(),
                name: tool_name.into(),
                arguments: arguments.into(),
            }]),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            name: None,
            content: content.into(),
        }
    }

    pub fn tool_with_name(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            name: Some(name.into()),
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_user() {
        let msg = ChatMessage::user("hello");
        match &msg {
            ChatMessage::User {
                content,
                images,
                ephemeral,
            } => {
                assert_eq!(content, "hello");
                assert!(images.is_empty());
                assert!(!ephemeral);
            }
            _ => panic!("Expected User variant"),
        }
    }

    #[test]
    fn chat_message_system() {
        let msg = ChatMessage::system("sys");
        assert!(!msg.is_ephemeral());
    }

    #[test]
    fn chat_message_system_ephemeral() {
        let msg = ChatMessage::system_ephemeral("sys");
        assert!(msg.is_ephemeral());
    }

    #[test]
    fn chat_message_user_ephemeral() {
        let msg = ChatMessage::user_ephemeral("hi");
        assert!(msg.is_ephemeral());
    }

    #[test]
    fn chat_message_assistant() {
        let msg = ChatMessage::assistant("response");
        match &msg {
            ChatMessage::Assistant { content, .. } => {
                assert_eq!(content.as_deref(), Some("response"));
            }
            _ => panic!("Expected Assistant variant"),
        }
    }

    #[test]
    fn chat_message_tool_call() {
        let msg = ChatMessage::assistant_tool_call("id1", "echo", r#"{"x":1}"#);
        match &msg {
            ChatMessage::Assistant { tool_calls, .. } => {
                let tc = tool_calls.as_ref().unwrap();
                assert_eq!(tc.len(), 1);
                assert_eq!(tc[0].id, "id1");
                assert_eq!(tc[0].name, "echo");
            }
            _ => panic!("Expected Assistant variant"),
        }
    }

    #[test]
    fn chat_message_tool() {
        let msg = ChatMessage::tool("id1", "result");
        match &msg {
            ChatMessage::Tool {
                tool_call_id,
                name,
                content,
            } => {
                assert_eq!(tool_call_id, "id1");
                assert!(name.is_none());
                assert_eq!(content, "result");
            }
            _ => panic!("Expected Tool variant"),
        }
    }

    #[test]
    fn chat_message_tool_with_name() {
        let msg = ChatMessage::tool_with_name("id1", "echo", "result");
        match &msg {
            ChatMessage::Tool {
                tool_call_id,
                name,
                content,
            } => {
                assert_eq!(tool_call_id, "id1");
                assert_eq!(name.as_deref(), Some("echo"));
                assert_eq!(content, "result");
            }
            _ => panic!("Expected Tool variant"),
        }
    }

    #[test]
    fn custom_message_not_ephemeral() {
        let msg = ChatMessage::Custom {
            role: "artifact".to_string(),
            data: serde_json::json!({}),
        };
        assert!(!msg.is_ephemeral());
    }

    #[test]
    fn serialization_roundtrip() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&json).unwrap();
        match deserialized {
            ChatMessage::User { content, .. } => assert_eq!(content, "hello"),
            _ => panic!("Expected User"),
        }
    }
}
