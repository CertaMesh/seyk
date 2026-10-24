#![cfg(feature = "onnx")]

//! Retrieval quality with the real model. Assertions are about ranking only.

mod model_support;

use std::fs;
use std::path::Path;

use seyk::{Dimension, MemoryCache, ResourceLocator, SearchEngine, SearchResults};

fn write(dir: &Path, name: &str, body: &str) {
    let p = dir.join(name);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

// These examples are invented for this test and use no private source text.
fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(
        d,
        "astronomy/telescope.md",
        "# Night sky\n\nPoint the telescope toward Saturn after sunset. Its rings become visible \
         through the eyepiece. A camera records the distant planets and stars.\n",
    );
    write(
        d,
        "kitchen/soup.txt",
        "Roast the tomatoes with garlic until the skins split. Add stock, simmer twenty minutes, \
         blend with a handful of basil and finish with olive oil.\n",
    );
    write(
        d,
        "tech/caching.md",
        "## Hot keys\n\nRedis keeps frequently read keys in memory with an expiry, so repeated lookups \
         skip the database entirely. Invalidate on write to avoid serving stale entries.\n",
    );
    write(
        d,
        "ideas/garden.md",
        "# Projects\n\n## Tools\n\nKeep the spade and rake in the shed.\n\n\
         ## Seedlings\n\nPlace the young plants near a bright window. Water their roots gently. \
         Move them outside only after the last frost.\n",
    );
    write(
        d,
        "workshop/clay.md",
        "Shape the wet clay into a bowl on the wheel. Let the pottery dry before firing it \
         in the kiln. The heat hardens the ceramic, ready for glazing.\n",
    );
    dir
}

fn search(engine: &SearchEngine, dir: &Path, query: &str) -> SearchResults {
    engine
        .search(query)
        .paths([dir])
        .limit(5)
        .execute()
        .unwrap()
}

fn top_file(r: &SearchResults) -> String {
    r.hits[0]
        .resource
        .to_file_path()
        .expect("file resource")
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn queries_without_shared_keywords_find_the_right_note() {
    let dir = corpus();
    let engine = SearchEngine::builder()
        .embedder(model_support::embedder(Dimension::D256))
        .cache(MemoryCache::new())
        .build()
        .unwrap();

    let cases = [
        ("observing distant planets", "telescope.md"),
        ("Which notes concern looking into space?", "telescope.md"),
        ("what can I cook for dinner tonight", "soup.txt"),
        ("how to make a slow service respond faster", "caching.md"),
        (
            "caring for young plants before moving them outdoors",
            "garden.md",
        ),
        ("baking ceramic bowls at high temperatures", "clay.md"),
    ];
    for (query, expected) in cases {
        let r = search(&engine, dir.path(), query);
        assert_eq!(top_file(&r), expected, "query {query:?}: {:#?}", r.hits);
    }
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn locator_points_at_the_matching_section() {
    let dir = corpus();
    let engine = SearchEngine::builder()
        .embedder(model_support::embedder(Dimension::D256))
        .build()
        .unwrap();
    let r = search(
        &engine,
        dir.path(),
        "caring for young plants before moving them outdoors",
    );
    let hit = &r.hits[0];
    assert_eq!(top_file(&r), "garden.md");
    assert_eq!(hit.heading.as_deref(), Some("Projects > Seedlings"));
    let ResourceLocator::Text {
        byte_start,
        byte_end,
        line_start,
        ..
    } = hit.locator
    else {
        panic!()
    };
    let source = fs::read_to_string(hit.resource.to_file_path().expect("file resource")).unwrap();
    let text = &source[byte_start as usize..byte_end as usize];
    assert!(text.starts_with("## Seedlings"), "{text:?}");
    assert_eq!(line_start, Some(7));
    assert!(
        hit.preview
            .as_deref()
            .unwrap()
            .starts_with("Place the young")
    );
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn ranking_holds_at_every_dimension() {
    let dir = corpus();
    for dim in Dimension::ALL {
        let engine = SearchEngine::builder()
            .embedder(model_support::embedder(dim))
            .build()
            .unwrap();
        let r = search(&engine, dir.path(), "observing distant planets");
        assert_eq!(top_file(&r), "telescope.md", "dimension {dim}");
    }
}
