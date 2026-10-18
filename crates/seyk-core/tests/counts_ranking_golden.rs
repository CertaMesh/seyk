mod common;
#[path = "support/golden_provider.rs"]
mod golden_provider;
use golden_provider::GoldenProvider;
use seyk_core::discovery::ResourceKind;
use seyk_core::segment::TextFormat;
use seyk_core::{Dimension, SearchEngine, SearchMode};

// Originally captured from main 28814a85; refreshed for the intentional RRF k=5 change.
// Stable fixture URIs exclude temporary paths; mtimes are normalized; score bits and complete
// locators, signals, ordering, headings and previews remain part of the golden.
#[test]
fn counts_preserve_rrf_k5_ranking_bytes() {
    let provider = GoldenProvider::new(
        [
            (
                "father.md",
                "# Family\nHe repeats the gestures he swore he never would.",
            ),
            (
                "kitchen.txt",
                "My father painted the kitchen yellow in spring.",
            ),
            (
                "garden.md",
                "Tomatoes need sun and a little basil beside them.",
            ),
            (
                "incident.txt",
                "Outage review INC-18372: the restore path never ran.",
            ),
        ]
        .map(|(name, text)| {
            let format = if name.ends_with(".md") {
                TextFormat::Markdown
            } else {
                TextFormat::Plain
            };
            (name, ResourceKind::Text(format), text.as_bytes().to_vec())
        }),
    );
    let engine = SearchEngine::builder()
        .provider(provider)
        .embedder(common::FakeEmbedder::new(Dimension::D256, "r1"))
        .build()
        .unwrap();
    let mut rows = Vec::new();
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Semantic,
        SearchMode::Lexical,
    ] {
        for query in ["father", "kitchen father", "INC-18372", "basil", "absent"] {
            for limit in [0, 1, 3, 20] {
                let result = engine
                    .search(query)
                    .query("family spring")
                    .roots([golden_provider::root()])
                    .mode(mode)
                    .limit(limit)
                    .kind_counts(true)
                    .execute()
                    .unwrap();
                let hits: Vec<_> = result
                    .hits
                    .into_iter()
                    .map(|mut h| {
                        h.metadata.modified_ns = None;
                        serde_json::json!({"score_bits": h.score.to_bits(), "hit":h})
                    })
                    .collect();
                rows.push(serde_json::json!({"mode":mode,"query":query,"limit":limit,"hits":hits}));
            }
        }
    }
    let actual = serde_json::to_string_pretty(&rows).unwrap() + "\n";
    assert_eq!(actual, include_str!("fixtures/counts-rrf-k5.json"));
}
