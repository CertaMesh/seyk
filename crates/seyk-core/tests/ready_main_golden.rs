//! Captured from main 71f9a18443483933f9bb5cf7f5fc18868fae4790 using this
//! fixture and FakeEmbedder. Only the temporary root, source mtimes, and wall
//! time are normalized. Ranking, scores, signals, locators, hashes, previews,
//! and deterministic stats are compared to pre-change output in all modes.
mod common;

use seyk_core::{Dimension, SearchEngine, SearchMode};

fn normalize(value: &mut serde_json::Value, root: &str) {
    match value {
        serde_json::Value::String(s) => *s = s.replace(root, "<CORPUS>"),
        serde_json::Value::Array(values) => {
            for value in values {
                normalize(value, root);
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
                    normalize(value, root);
                }
            }
        }
        _ => {}
    }
}

#[test]
fn ready_results_match_main_71f9a18_golden() {
    let root = tempfile::TempDir::new().unwrap();
    for (name, text) in [
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
    ] {
        std::fs::write(root.path().join(name), text).unwrap();
    }
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D256, "r1"))
        .build()
        .unwrap();
    engine.index().paths([root.path()]).execute().unwrap();
    let mut actual = Vec::new();
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Lexical,
        SearchMode::Semantic,
    ] {
        let results = engine
            .search("father")
            .query("INC-18372")
            .paths([root.path()])
            .mode(mode)
            .sync(false)
            .lexical_fallback(true)
            .execute()
            .unwrap();
        let mut value = serde_json::to_value(results).unwrap();
        normalize(
            &mut value,
            root.path().canonicalize().unwrap().to_str().unwrap(),
        );
        actual.push(value);
    }
    let actual = serde_json::to_string_pretty(&actual).unwrap() + "\n";
    assert_eq!(actual, include_str!("fixtures/ready-main-71f9a18.json"));
}
