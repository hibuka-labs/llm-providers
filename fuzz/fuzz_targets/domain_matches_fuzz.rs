#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &str| {
    if let Some(idx) = data.find('\0') {
        let url = &data[..idx];
        let domain = &data[idx + 1..];
        let _ = llm_unified::fuzz::model_registry::domain_matches(url, domain);
    }
});
