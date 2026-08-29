//! Usage info types.

/// Token usage information for an LLM call.
#[derive(Clone, Debug, Default)]
pub struct UsageInfo {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    /// Tokens used for reasoning / thinking (DeepSeek, OpenAI o-series, etc.).
    /// Always `None` on the Anthropic protocol: its usage object has no
    /// thinking breakdown — thinking tokens are folded into
    /// `completion_tokens`.
    pub reasoning_tokens: Option<u32>,
}

impl UsageInfo {
    /// Merge a partial usage event into `self`; the incoming `Some` field
    /// wins over the existing value, a `None` leaves it untouched.
    ///
    /// Streaming protocols may split usage across several events (Anthropic
    /// sends `prompt_tokens` in `message_start` and the final
    /// `completion_tokens` in `message_delta`). Replacing the whole struct
    /// on each event would silently zero out fields the last event omits,
    /// so consumers must fold events with this method instead.
    pub fn merge(&mut self, other: &UsageInfo) {
        if other.prompt_tokens.is_some() {
            self.prompt_tokens = other.prompt_tokens;
        }
        if other.completion_tokens.is_some() {
            self.completion_tokens = other.completion_tokens;
        }
        if other.total_tokens.is_some() {
            self.total_tokens = other.total_tokens;
        }
        if other.reasoning_tokens.is_some() {
            self.reasoning_tokens = other.reasoning_tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_keeps_existing_when_incoming_none() {
        let mut acc = UsageInfo {
            prompt_tokens: Some(5),
            completion_tokens: Some(2),
            total_tokens: Some(7),
            reasoning_tokens: None,
        };
        // Anthropic `message_delta`: only the final output count arrives.
        acc.merge(&UsageInfo {
            prompt_tokens: None,
            completion_tokens: Some(26),
            total_tokens: None,
            reasoning_tokens: None,
        });
        assert_eq!(
            acc.prompt_tokens,
            Some(5),
            "prompt survives the partial event"
        );
        assert_eq!(acc.completion_tokens, Some(26), "later Some wins");
        assert_eq!(acc.total_tokens, Some(7));
    }

    #[test]
    fn merge_lets_later_some_override() {
        // DashScope's anthropic endpoint reports the full `input_tokens` in
        // `message_delta` (context so far), superseding the `message_start`
        // value (latest message only) — the newer number is more useful.
        let mut acc = UsageInfo {
            prompt_tokens: Some(2),
            ..Default::default()
        };
        acc.merge(&UsageInfo {
            prompt_tokens: Some(63),
            ..Default::default()
        });
        assert_eq!(acc.prompt_tokens, Some(63));
    }
}
