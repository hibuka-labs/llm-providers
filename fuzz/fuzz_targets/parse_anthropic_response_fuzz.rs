#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = llm_unified::fuzz::anthropic::parse_anthropic_response(data);
});
