mod common;
use seyk_core::{Dimension, SearchEngine, SearchMode};

// Captured from origin/main 28814a85, before adding count/progress code.
// Only temporary paths and mtimes are normalized; score bits and complete
// locators, signals, ordering, headings and previews remain part of the golden.
#[test]
fn counts_preserve_main_ranking_bytes() {
    let dir = tempfile::tempdir().unwrap();
    for (name, text) in [
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
    ] {
        std::fs::write(dir.path().join(name), text).unwrap();
    }
    let engine = SearchEngine::builder()
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
                    .paths([dir.path()])
                    .mode(mode)
                    .limit(limit)
                    .kind_counts(true)
                    .execute()
                    .unwrap();
                let hits: Vec<_> = result
                    .hits
                    .into_iter()
                    .map(|mut h| {
                        let name = h
                            .resource
                            .to_file_path()
                            .unwrap()
                            .file_name()
                            .unwrap()
                            .to_owned();
                        h.resource = format!("mem://golden/{}", name.to_string_lossy())
                            .parse()
                            .unwrap();
                        h.metadata.modified_ns = None;
                        serde_json::json!({"score_bits": h.score.to_bits(), "hit":h})
                    })
                    .collect();
                rows.push(serde_json::json!({"mode":mode,"query":query,"limit":limit,"hits":hits}));
            }
        }
    }
    let actual = serde_json::to_string_pretty(&rows).unwrap() + "\n";
    assert_eq!(actual, include_str!("fixtures/counts-main-ranking.json"));
}
