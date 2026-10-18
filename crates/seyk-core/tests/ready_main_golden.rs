//! Originally captured from main 71f9a18; refreshed for RRF k=5 using this
//! fixture and FakeEmbedder. Stable URIs map to the historical root; source mtimes and wall
//! time are normalized. Ranking, scores, signals, locators, hashes, previews,
//! and deterministic stats are compared to the k=5 output in all modes.
mod common;
#[path = "support/golden_provider.rs"]
mod golden_provider;
use golden_provider::GoldenProvider;
use seyk_core::discovery::ResourceKind;
use seyk_core::segment::TextFormat;

use seyk_core::{Dimension, SearchEngine, SearchMode};

fn normalize(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(s) => {
            let historical = match s.as_str() {
                "mem://golden/father.md" => "file://<CORPUS>/father.md",
                "mem://golden/kitchen.md" => "file://<CORPUS>/kitchen.md",
                "mem://golden/incident.md" => "file://<CORPUS>/incident.md",
                "mem://golden/garden.md" => "file://<CORPUS>/garden.md",
                _ => return,
            };
            *s = historical.into();
        }
        serde_json::Value::Array(values) => {
            for value in values {
                normalize(value);
            }
        }
        serde_json::Value::Object(object) => {
            // Main added provenance after this historical golden was captured.
            // Its complete value is covered by counts_ranking_golden and hybrid.
            if object.contains_key("signals") && object.contains_key("locator") {
                object.remove("match");
            }
            for (key, value) in object {
                if key == "modified_ns" || key.ends_with("_ms") {
                    *value = serde_json::json!(0);
                } else {
                    normalize(value);
                }
            }
        }
        _ => {}
    }
}

#[test]
fn ready_results_match_rrf_k5_golden() {
    let provider = GoldenProvider::new(
        [
            (
                "father.md",
                "He repeats the gestures he swore he never would.",
            ),
            (
                "kitchen.md",
                "My father painted the kitchen yellow in spring.",
            ),
            (
                "incident.md",
                "Outage review for INC-18372: the restore path never ran.",
            ),
            (
                "garden.md",
                "Tomatoes need sun and a little basil beside them.",
            ),
        ]
        .map(|(name, text)| {
            (
                name,
                ResourceKind::Text(TextFormat::Markdown),
                text.as_bytes().to_vec(),
            )
        }),
    );
    let engine = SearchEngine::builder()
        .provider(provider)
        .embedder(common::FakeEmbedder::new(Dimension::D256, "r1"))
        .build()
        .unwrap();
    engine
        .index()
        .roots([golden_provider::root()])
        .execute()
        .unwrap();
    let mut actual = Vec::new();
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Lexical,
        SearchMode::Semantic,
    ] {
        let results = engine
            .search("father")
            .query("INC-18372")
            .roots([golden_provider::root()])
            .mode(mode)
            .sync(false)
            .lexical_fallback(true)
            .execute()
            .unwrap();
        let mut value = serde_json::to_value(results).unwrap();
        normalize(&mut value);
        actual.push(value);
    }
    let actual = serde_json::to_string_pretty(&actual).unwrap() + "\n";
    assert_eq!(actual, include_str!("fixtures/ready-rrf-k5.json"));
}
