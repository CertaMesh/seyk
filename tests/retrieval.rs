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

fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(
        d,
        "characters/martin.md",
        "# Martin\n\nLately Martin catches himself clearing his throat before he speaks, the way the old man did. \
         He hates it. Every time his voice hardens with the children he hears the house he grew up in, \
         and he wonders whether the anger was inherited like the shape of his hands.\n",
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
        "ideas/memory.md",
        "# Ideas\n\n## Weather\n\nA chapter that opens with rain on the allotment.\n\n\
         ## Remembering wrong\n\nShe is certain the argument happened in winter, but the photographs show summer. \
         Each time she retells it, the story shifts a little, and she can no longer tell which version is hers.\n",
    );
    write(
        d,
        "scenes/funeral.md",
        "After the funeral her chest felt packed with wet sand. She could not lift her arms to hang up her coat; \
         the loss sat in her shoulders like a yoke.\n",
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
        (
            "Where did I write about the protagonist becoming like his father?",
            "martin.md",
        ),
        (
            "Which notes deal with somebody repeating their parents' behaviour?",
            "martin.md",
        ),
        ("what can I cook for dinner tonight", "soup.txt"),
        ("how to make a slow service respond faster", "caching.md"),
        (
            "What ideas did I have about memory becoming unreliable?",
            "memory.md",
        ),
        (
            "Where did I describe grief as something physical?",
            "funeral.md",
        ),
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
        "misremembering when an argument took place",
    );
    let hit = &r.hits[0];
    assert_eq!(top_file(&r), "memory.md");
    assert_eq!(hit.heading.as_deref(), Some("Ideas > Remembering wrong"));
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
    assert!(text.starts_with("## Remembering wrong"), "{text:?}");
    assert_eq!(line_start, Some(7));
    assert!(
        hit.preview
            .as_deref()
            .unwrap()
            .starts_with("She is certain")
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
        let r = search(&engine, dir.path(), "fear of turning into your dad");
        assert_eq!(top_file(&r), "martin.md", "dimension {dim}");
    }
}
