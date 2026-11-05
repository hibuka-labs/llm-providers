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

/// `skip_serializing_if` helper for `bool` flags: a message that is not
/// synthetic serializes exactly as it did before the flag existed.
fn is_false(value: &bool) -> bool {
    !*value
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
        /// Machine-composed input on the user role (tool-failure notices,
        /// background-task wakes, the window seed): the model sees it like
        /// any user message, but it is never the user's own words. Context
        /// rotation skips synthetic messages when carrying the task trail
        /// into a fresh window. Unlike `ephemeral`, this flag survives
        /// persistence, so a resumed session keeps the distinction.
        #[serde(default, skip_serializing_if = "is_false")]
        synthetic: bool,
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
    /// A tool's result, paired with the assistant's `tool_calls` entry.
    ///
    /// `content` is the textual face; `images` carries pictures the tool's
    /// output wants the model to *look at* (a screenshot, a rendered
    /// diagram). A provider whose tool-result shape can carry images sends
    /// them beside the text; one that cannot says so in `content` rather
    /// than dropping the picture silently.
    Tool {
        tool_call_id: String,
        name: Option<String>,
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageAttachment>,
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
            synthetic: false,
        }
    }

    /// Create an ephemeral user message: auto-cleaned after turn, not persisted.
    pub fn user_ephemeral(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
            ephemeral: true,
            synthetic: false,
        }
    }

    /// Create a synthetic user message: machine-composed input on the user
    /// role (a notice, a wake, a seed) that must never stand in for the
    /// user's own words — context rotation skips it when collecting the task
    /// trail. The flag is persisted, so a resumed session keeps the
    /// distinction.
    pub fn user_synthetic(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
            ephemeral: false,
            synthetic: true,
        }
    }

    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self::User {
            content: content.into(),
            images,
            ephemeral: false,
            synthetic: false,
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

    /// Whether this is machine-composed input on the user role (never the
    /// user's own words).
    pub fn is_synthetic(&self) -> bool {
        match self {
            Self::User { synthetic, .. } => *synthetic,
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
            images: Vec::new(),
        }
    }

    /// A tool result that also hands the model pictures to look at.
    pub fn tool_with_images(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        images: Vec<ImageAttachment>,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            name: None,
            content: content.into(),
            images,
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
            images: Vec::new(),
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
                synthetic,
            } => {
                assert_eq!(content, "hello");
                assert!(images.is_empty());
                assert!(!ephemeral);
                assert!(!synthetic);
            }
            _ => panic!("Expected User variant"),
        }
    }

    #[test]
    fn chat_message_user_synthetic() {
        let msg = ChatMessage::user_synthetic("Background tasks finished");
        assert!(msg.is_synthetic());
        assert!(!msg.is_ephemeral());
        // A plain user message is the user's own words.
        assert!(!ChatMessage::user("hello").is_synthetic());
    }

    /// The synthetic flag survives persistence — a resumed session must keep
    /// the distinction between the user's words and machine notices.
    #[test]
    fn a_synthetic_user_message_round_trips_its_flag() {
        let wire =
            serde_json::to_string(&ChatMessage::user_synthetic("Tool calls failed")).unwrap();
        assert!(wire.contains("synthetic"), "flag must persist: {wire}");
        let back: ChatMessage = serde_json::from_str(&wire).unwrap();
        assert!(back.is_synthetic());
    }

    /// An all-real transcript serializes exactly as it did before the flag
    /// existed, and history written before it still loads as real.
    #[test]
    fn a_real_user_message_serializes_without_a_synthetic_key() {
        let wire = serde_json::to_string(&ChatMessage::user("hello")).unwrap();
        assert!(
            !wire.contains("synthetic"),
            "no empty flag on the wire: {wire}"
        );
        let old = r#"{"User":{"content":"hello","images":[]}}"#;
        let back: ChatMessage = serde_json::from_str(old).unwrap();
        assert!(!back.is_synthetic());
    }

    #[test]
    fn chat_message_system() {
        let msg = ChatMessage::system("sys");
        assert!(!msg.is_ephemeral());
    }

    /// Pictures ride the tool result through a persisted transcript too —
    /// a session resumed later still hands the model what it was shown.
    #[test]
    fn a_tool_result_round_trips_its_pictures() {
        let msg = ChatMessage::tool_with_images(
            "tc1",
            "image: the whole desktop",
            vec![ImageAttachment::Base64 {
                data: "aGVsbG8=".to_string(),
                media_type: Some("image/png".to_string()),
                detail: None,
            }],
        );
        let wire = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&wire).unwrap();
        match back {
            ChatMessage::Tool {
                tool_call_id,
                content,
                images,
                ..
            } => {
                assert_eq!(tool_call_id, "tc1");
                assert_eq!(content, "image: the whole desktop");
                assert_eq!(images.len(), 1);
            }
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    /// Transcripts written before the image face existed have no `images`
    /// key at all. They must keep loading — a resume must never fail on
    /// history the product itself wrote.
    #[test]
    fn a_tool_result_written_before_images_still_loads() {
        let old = r#"{"Tool":{"tool_call_id":"tc1","name":null,"content":"done"}}"#;
        let msg: ChatMessage = serde_json::from_str(old).unwrap();
        match msg {
            ChatMessage::Tool {
                content, images, ..
            } => {
                assert_eq!(content, "done");
                assert!(images.is_empty());
            }
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    /// An empty image list is not written at all: a word-only tool result
    /// serializes byte-for-byte as it did before this field existed.
    #[test]
    fn a_word_only_tool_result_serializes_without_an_images_key() {
        let wire = serde_json::to_string(&ChatMessage::tool("tc1", "done")).unwrap();
        assert!(
            !wire.contains("images"),
            "no empty field on the wire: {wire}"
        );
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
                images,
            } => {
                assert_eq!(tool_call_id, "id1");
                assert!(name.is_none());
                assert_eq!(content, "result");
                assert!(
                    images.is_empty(),
                    "the plain constructor carries no pictures"
                );
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
                images,
            } => {
                assert_eq!(tool_call_id, "id1");
                assert_eq!(name.as_deref(), Some("echo"));
                assert_eq!(content, "result");
                assert!(
                    images.is_empty(),
                    "the plain constructor carries no pictures"
                );
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
