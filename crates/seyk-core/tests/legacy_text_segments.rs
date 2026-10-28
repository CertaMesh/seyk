use serde::Deserialize;
use seyk_core::segment::{TextFormat, TextSegment, TextSegmenter, TextSegmenterConfig};

#[derive(Deserialize)]
struct Golden {
    base: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    format: TextFormat,
    source: String,
    config: TextSegmenterConfig,
    version: String,
    segments: Vec<TextSegment>,
}

#[test]
fn legacy_plain_markdown_and_code_match_main_byte_for_byte() {
    let golden: Golden =
        serde_json::from_str(include_str!("fixtures/legacy-text-segments.json")).unwrap();
    assert_eq!(golden.base, "2e1a1077d78e1d12fc69f1a14b1d537a504aa225");
    assert_eq!(golden.cases.len(), 6);
    for case in golden.cases {
        let segmenter = TextSegmenter::new(case.config);
        assert_eq!(segmenter.version(), case.version);
        assert_eq!(segmenter.segment(&case.source, case.format), case.segments);
    }
}
