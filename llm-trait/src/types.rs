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

    /// Prompt tokens served from the provider's prefix cache, billed at the
    /// cache-read rate rather than the input rate. Anthropic reports
    /// `cache_read_input_tokens`; OpenAI reports
    /// `prompt_tokens_details.cached_tokens`; a protocol whose provider does
    /// not cache reports `None`. `None` means "unknown", never "zero": a cache
    /// miss and a counter that does not exist look the same from here.
    pub cache_read_tokens: Option<u32>,
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
        if other.cache_read_tokens.is_some() {
            self.cache_read_tokens = other.cache_read_tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Value: protects=cache hits survive streaming usage folds; fails_when=
    /// `merge` drops `cache_read_tokens` and the app cannot see what it saved.
    /// why_new=no test folds cache usage today — the field is new.
    /// seam=none; no_claim=that a given provider reports it (per protocol).
    #[test]
    fn merge_keeps_the_cache_hit_count() {
        let mut acc = UsageInfo {
            prompt_tokens: Some(40_000),
            cache_read_tokens: Some(38_000),
            ..Default::default()
        };
        // Anthropic `message_delta` carries only the output count — a `None`
        // must not erase the hit count that `message_start` reported.
        acc.merge(&UsageInfo {
            completion_tokens: Some(700),
            ..Default::default()
        });
        assert_eq!(acc.cache_read_tokens, Some(38_000));
        assert_eq!(acc.completion_tokens, Some(700));
    }

    #[test]
    fn merge_keeps_existing_when_incoming_none() {
        let mut acc = UsageInfo {
            prompt_tokens: Some(5),
            completion_tokens: Some(2),
            total_tokens: Some(7),
            reasoning_tokens: None,
            cache_read_tokens: None,
        };
        // Anthropic `message_delta`: only the final output count arrives.
        acc.merge(&UsageInfo {
            prompt_tokens: None,
            completion_tokens: Some(26),
            total_tokens: None,
            reasoning_tokens: None,
            cache_read_tokens: None,
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
