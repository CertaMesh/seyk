mod common;

use std::fs;

use common::FakeEmbedder;
use seyk_core::{Dimension, MemoryCache, SearchEngine, SearchMode};
use tempfile::TempDir;

#[test]
fn several_queries_keep_each_segments_best_score() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("garden.md"), "alpha garden").unwrap();
    fs::write(temp.path().join("river.md"), "beta river").unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D256, "r1"))
        .cache(MemoryCache::new())
        .build()
        .unwrap();
    let file_of = |h: &seyk_core::SearchHit| {
        h.resource
            .to_file_path()
            .expect("file resource")
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    };

    let single = engine
        .search("beta river")
        .mode(SearchMode::Semantic)
        .paths([temp.path()])
        .limit(1)
        .execute()
        .unwrap();
    let both = engine
        .search("alpha garden")
        .query("beta river")
        .mode(SearchMode::Semantic)
        .paths([temp.path()])
        .limit(2)
        .execute()
        .unwrap();

    assert_eq!(both.queries, ["alpha garden", "beta river"]);
    let river = both.hits.iter().find(|h| file_of(h) == "river.md").unwrap();
    let garden = both
        .hits
        .iter()
        .find(|h| file_of(h) == "garden.md")
        .unwrap();
    assert_eq!(river.query_index, 1);
    assert_eq!(garden.query_index, 0);
    // Fused scores are the plain cosine of the best query, not a blend.
    assert_eq!(river.score, single.hits[0].score);
}
