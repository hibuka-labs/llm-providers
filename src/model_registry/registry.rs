//! Model Registry — centralized model knowledge.
//!
//! Each brand has its own module with profile data and brand prefix rules.
//! The `ModelRegistry` collects all data and provides `lookup()` for the factory.

use std::collections::HashMap;

use llm_trait::{Capabilities, Protocol, ReasoningMode};

/// Model capability profile (pure data, no behavior).
///
/// Each profile describes one model × protocol combination.
/// The same model can have different profiles for different protocols.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Wire protocol
    pub protocol: Protocol,
    /// Provider name (for `info().name`)
    pub provider_name: &'static str,
    /// Capabilities (supports_thinking, supports_tools, etc.)
    pub capabilities: Capabilities,
    /// Reasoning mode (how to express reasoning: Effort / Thinking / None)
    pub reasoning_mode: ReasoningMode,
    /// Supported extra request params (e.g., `reasoning_effort`)
    /// Params not in this list are silently ignored for safety.
    pub supported_extra_params: &'static [&'static str],
}

/// Global model registry.
///
/// Profile key format: `model@protocol` (composite key).
/// For example, "mimo-v2.5-pro@openai" and "mimo-v2.5-pro@anthropic"
/// are two independent profiles.
pub struct ModelRegistry {
    /// Profile storage, key = "model@protocol"
    profiles: HashMap<String, ModelProfile>,
    /// Brand inference: model prefix → brand name
    /// E.g., "mimo-" → "mimo", "gpt-" → "gpt"
    brand_prefixes: Vec<(&'static str, &'static str)>,
}

impl ModelRegistry {
    /// Assemble data from all model modules.
    pub fn builtin() -> Self {
        let mut profiles = HashMap::new();
        let mut brand_prefixes = Vec::new();

        for (name, profile) in super::gpt::profiles() {
            profiles.insert(name.to_string(), profile);
        }
        for (name, profile) in super::anthropic::profiles() {
            profiles.insert(name.to_string(), profile);
        }
        for (name, profile) in super::deepseek::profiles() {
            profiles.insert(name.to_string(), profile);
        }
        for (name, profile) in super::mimo::profiles() {
            profiles.insert(name.to_string(), profile);
        }
        for (name, profile) in super::qwen::profiles() {
            profiles.insert(name.to_string(), profile);
        }

        brand_prefixes.extend(super::gpt::brand_prefixes());
        brand_prefixes.extend(super::anthropic::brand_prefixes());
        brand_prefixes.extend(super::deepseek::brand_prefixes());
        brand_prefixes.extend(super::mimo::brand_prefixes());
        brand_prefixes.extend(super::qwen::brand_prefixes());

        Self {
            profiles,
            brand_prefixes,
        }
    }

    /// Lookup model profile with fallback chain:
    /// 1. Determine protocol (explicit > URL inference > default OpenAI)
    /// 2. Exact match "model@protocol"
    /// 3. Brand default "brand@protocol"
    /// 4. Default OpenAI safe profile
    pub fn lookup(
        &self,
        model: &str,
        base_url: Option<&str>,
        explicit_protocol: Option<Protocol>,
    ) -> ModelProfile {
        // 1. Determine protocol
        let protocol = explicit_protocol
            .or_else(|| self.infer_protocol_from_url(model, base_url))
            .unwrap_or(Protocol::OpenAi);

        // 2. Exact match "model@protocol"
        let exact_key = format!("{}@{}", model, protocol.as_str());
        if let Some(profile) = self.profiles.get(exact_key.as_str()) {
            tracing::debug!(
                model,
                protocol = ?protocol,
                matched = %exact_key,
                "registry: exact match"
            );
            return profile.clone();
        }

        // 3. Brand default "brand@protocol"
        if let Some(brand) = self.infer_brand(model) {
            let brand_key = format!("{}@{}", brand, protocol.as_str());
            if let Some(profile) = self.profiles.get(brand_key.as_str()) {
                tracing::debug!(
                    model,
                    brand,
                    protocol = ?protocol,
                    matched = %brand_key,
                    "registry: brand match"
                );
                return profile.clone();
            }
        }

        // 4. Default profile for the determined protocol
        tracing::debug!(
            model,
            protocol = ?protocol,
            "registry: using default fallback profile"
        );
        self.default_profile(protocol)
    }

    /// Infer brand from model name using brand_prefixes.
    fn infer_brand(&self, model: &str) -> Option<&str> {
        for (prefix, brand) in &self.brand_prefixes {
            if model.starts_with(*prefix) {
                return Some(brand);
            }
        }
        None
    }

    /// Infer protocol from URL.
    ///
    /// Detection rules (in order):
    /// 1. Domain contains "anthropic.com" → Anthropic (official API)
    /// 2. Path contains "/anthropic" → Anthropic (third-party providers)
    /// 3. Otherwise → None (will use explicit protocol or default OpenAI)
    fn infer_protocol_from_url(&self, _model: &str, base_url: Option<&str>) -> Option<Protocol> {
        let url = base_url?;

        // Official Anthropic API: https://api.anthropic.com
        // Use domain-boundary matching to avoid false positives like "notanthropic.com"
        if Self::domain_matches(url, "anthropic.com") {
            return Some(Protocol::Anthropic);
        }

        // Universal: URL path (after host) contains "/anthropic" → Anthropic protocol
        // Matches: /anthropic, /apps/anthropic, /v1/anthropic, etc.
        // Must check only the path portion to avoid matching hostnames like "anthropic.com.evil.com"
        if let Some(scheme_end) = url.find("://") {
            let after_scheme = &url[scheme_end + 3..];
            if let Some(path_start) = after_scheme.find('/') {
                let path = &after_scheme[path_start..];
                if path.contains("/anthropic") {
                    return Some(Protocol::Anthropic);
                }
            }
        }

        None
    }

    /// Check if a URL's host is or ends with the given domain.
    ///
    /// Matches:
    /// - `https://anthropic.com/...` (exact domain)
    /// - `https://api.anthropic.com/...` (subdomain)
    ///
    /// Rejects:
    /// - `https://notanthropic.com/...` (different domain)
    /// - `https://anthropic.com.evil.com/...` (domain is only a prefix)
    fn domain_matches(url: &str, domain: &str) -> bool {
        // Extract the host portion from the URL (between `://` and first `/`, `:`, `?`, `#`)
        let host_start = url.find("://").map(|p| p + 3);
        let host_start = match host_start {
            Some(s) => s,
            None => return false,
        };
        let rest = &url[host_start..];
        let host_end = rest
            .find(|c: char| c == '/' || c == ':' || c == '?' || c == '#')
            .unwrap_or(rest.len());
        let host = &rest[..host_end];

        // Exact match: host == domain
        if host == domain {
            return true;
        }

        // Subdomain match: host ends with `.domain`
        host.ends_with(&format!(".{}", domain))
    }

    fn default_profile(&self, protocol: Protocol) -> ModelProfile {
        match protocol {
            Protocol::Anthropic => ModelProfile {
                protocol: Protocol::Anthropic,
                provider_name: "anthropic",
                capabilities: Capabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    supports_thinking: true,
                    max_context_tokens: Some(200_000),
                    max_output_tokens: Some(8_192),
                },
                reasoning_mode: ReasoningMode::Thinking,
                supported_extra_params: &[],
            },
            _ => ModelProfile {
                protocol: Protocol::OpenAi,
                provider_name: "openai",
                capabilities: Capabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: false,
                    supports_thinking: false,
                    max_context_tokens: Some(128_000),
                    max_output_tokens: Some(16_384),
                },
                reasoning_mode: ReasoningMode::None,
                supported_extra_params: &[],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_exact_model() {
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup("deepseek-chat", Some("https://api.deepseek.com/v1"), None);
        assert_eq!(profile.provider_name, "deepseek");
        assert_eq!(profile.protocol, Protocol::OpenAi);
    }

    #[test]
    fn lookup_brand_fallback() {
        let registry = ModelRegistry::builtin();
        // deepseek-xxx not registered, falls back to deepseek@openai
        let profile = registry.lookup("deepseek-xxx", Some("https://api.deepseek.com/v1"), None);
        assert_eq!(profile.provider_name, "deepseek");
    }

    #[test]
    fn lookup_mimo_openai_no_reasoning() {
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "mimo-v2.5-pro",
            Some("https://token-plan-cn.xiaomimimo.com/v1"),
            None,
        );
        assert_eq!(profile.provider_name, "mimo");
        assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    }

    #[test]
    fn lookup_unknown_model_default() {
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup("some-unknown", Some("https://api.example.com/v1"), None);
        assert_eq!(profile.provider_name, "openai");
        assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    }

    #[test]
    fn lookup_unknown_model_capabilities_are_false() {
        // P1-7 FIXED: Default OpenAI fallback now reports reasonable capabilities.
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup("some-unknown", Some("https://api.example.com/v1"), None);
        assert!(
            profile.capabilities.supports_streaming,
            "P1-7 FIXED: default fallback should support streaming"
        );
        assert!(
            profile.capabilities.supports_tools,
            "P1-7 FIXED: default fallback should support tools"
        );
        assert_eq!(
            profile.capabilities.max_output_tokens,
            Some(16_384),
            "P1-7 FIXED: default fallback should have max_output_tokens"
        );
    }

    #[test]
    fn lookup_explicit_protocol() {
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "mimo-v2.5-pro",
            Some("https://api.example.com/v1"),
            Some(Protocol::Anthropic),
        );
        assert_eq!(profile.protocol, Protocol::Anthropic);
        assert_eq!(profile.reasoning_mode, ReasoningMode::Thinking);
    }

    #[test]
    fn url_inference_mimo_anthropic() {
        // MiMo: https://token-plan-cn.xiaomimimo.com/anthropic
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "mimo-v2.5-pro",
            Some("https://token-plan-cn.xiaomimimo.com/anthropic"),
            None,
        );
        assert_eq!(profile.protocol, Protocol::Anthropic);
        assert_eq!(profile.provider_name, "mimo");
        assert_eq!(profile.reasoning_mode, ReasoningMode::Thinking);
    }

    #[test]
    fn url_inference_deepseek_anthropic() {
        // DeepSeek: https://api.deepseek.com/anthropic
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "deepseek-chat",
            Some("https://api.deepseek.com/anthropic"),
            None,
        );
        assert_eq!(profile.protocol, Protocol::Anthropic);
    }

    #[test]
    fn url_inference_qwen_anthropic() {
        // Qwen: https://dashscope.aliyuncs.com/apps/anthropic
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "qwen-plus",
            Some("https://dashscope.aliyuncs.com/apps/anthropic"),
            None,
        );
        assert_eq!(profile.protocol, Protocol::Anthropic);
    }

    #[test]
    fn url_inference_qwen_token_plan_anthropic() {
        // Qwen Token Plan: https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "qwen-plus",
            Some("https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"),
            None,
        );
        assert_eq!(profile.protocol, Protocol::Anthropic);
    }

    #[test]
    fn url_inference_default_openai() {
        // No /anthropic path → default to OpenAI
        let registry = ModelRegistry::builtin();
        let profile = registry.lookup(
            "mimo-v2.5-pro",
            Some("https://token-plan-cn.xiaomimimo.com/v1"),
            None,
        );
        assert_eq!(profile.protocol, Protocol::OpenAi);
        assert_eq!(profile.reasoning_mode, ReasoningMode::None);
    }

    #[test]
    fn url_inference_fake_anthropic_domain_not_matched() {
        // P2-16: "anthropic.com" substring matching should not match fake domains
        let registry = ModelRegistry::builtin();

        // notanthropic.com should NOT match
        let profile = registry.lookup(
            "some-model",
            Some("https://notanthropic.com/v1"),
            None,
        );
        assert_eq!(
            profile.protocol,
            Protocol::OpenAi,
            "P2-16: notanthropic.com should not match Anthropic"
        );

        // anthropic.com.evil.com should NOT match
        let profile2 = registry.lookup(
            "some-model",
            Some("https://anthropic.com.evil.com/v1"),
            None,
        );
        assert_eq!(
            profile2.protocol,
            Protocol::OpenAi,
            "P2-16: anthropic.com.evil.com should not match Anthropic"
        );
    }

    // ── proptest: domain_matches ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn domain_matches_never_panics(
                url in r"https?://[a-zA-Z0-9._:/-]{0,100}",
                domain in r"[a-z]{1,20}(\.[a-z]{1,10}){0,3}",
            ) {
                let _ = ModelRegistry::domain_matches(&url, &domain);
            }

            #[test]
            fn domain_matches_known_domain(url_path in r"/[a-zA-Z0-9._:/-]{0,50}") {
                let url = format!("https://api.anthropic.com{}", url_path);
                assert!(ModelRegistry::domain_matches(&url, "anthropic.com"),
                    "api.anthropic.com should match anthropic.com, url={}", url);
            }

            #[test]
            fn domain_matches_exact_domain(url_path in r"/[a-zA-Z0-9._:/-]{0,50}") {
                let url = format!("https://anthropic.com{}", url_path);
                assert!(ModelRegistry::domain_matches(&url, "anthropic.com"),
                    "anthropic.com should match exactly, url={}", url);
            }

            #[test]
            fn domain_matches_rejects_prefix_domain(url_path in r"[a-zA-Z0-9._:/-]{0,50}") {
                let url = format!("https://notanthropic.com{}", url_path);
                assert!(!ModelRegistry::domain_matches(&url, "anthropic.com"),
                    "notanthropic.com should NOT match anthropic.com, url={}", url);
            }

            #[test]
            fn domain_matches_rejects_suffix_domain(url_path in r"[a-zA-Z0-9._:/-]{0,50}") {
                let url = format!("https://anthropic.com.evil.com{}", url_path);
                assert!(!ModelRegistry::domain_matches(&url, "anthropic.com"),
                    "anthropic.com.evil.com should NOT match anthropic.com, url={}", url);
            }
        }
    }
}
