mod common;

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::FakeEmbedder;
use seyk::{
    Dimension, Embedder, Embedding, EmbeddingCache, MemoryCache, Modality, ModelIdentity,
    ResourceInput, Result, SearchEngine, SearchMode, SearchResults, SqliteCache,
};
use tempfile::TempDir;

/// Counts query embeddings, to prove lexical mode never runs the model on one.
struct Counting {
    inner: FakeEmbedder,
    queries: AtomicUsize,
}

impl Embedder for Counting {
    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }
    fn supports(&self, m: Modality) -> bool {
        self.inner.supports(m)
    }
    fn embed_query(&self, q: &str) -> Result<Embedding> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.embed_query(q)
    }
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        self.inner.embed_resources(inputs)
    }
}

fn corpus() -> TempDir {
    let temp = TempDir::new().unwrap();
    let w = |name: &str, text: &str| fs::write(temp.path().join(name), text).unwrap();
    w(
        "father.md",
        "He repeats the gestures he swore he never would.",
    );
    w(
        "kitchen.md",
        "My father painted the kitchen yellow in spring.",
    );
    w(
        "incident.md",
        "Outage review for INC-18372: the restore path never ran.",
    );
    w(
        "garden.md",
        "Tomatoes need sun and a little basil beside them.",
    );
    temp
}

fn engine<C: EmbeddingCache + 'static>(cache: C) -> (SearchEngine, Arc<Counting>) {
    let embedder = Arc::new(Counting {
        inner: FakeEmbedder::new(Dimension::D256, "r1"),
        queries: AtomicUsize::new(0),
    });
    let engine = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .cache(cache)
        .build()
        .unwrap();
    (engine, embedder)
}

fn search(engine: &SearchEngine, root: &Path, q: &str, mode: SearchMode) -> SearchResults {
    engine.search(q).mode(mode).paths([root]).execute().unwrap()
}

fn top(r: &SearchResults) -> String {
    r.hits[0]
        .resource
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

fn both_caches(test: impl Fn(SearchEngine, Arc<Counting>, &Path)) {
    let temp = corpus();
    let (e, c) = engine(MemoryCache::new());
    test(e, c, temp.path());
    let db = TempDir::new().unwrap();
    let (e, c) = engine(SqliteCache::open(db.path().join("cache.sqlite3")).unwrap());
    test(e, c, temp.path());
}

#[test]
fn a_file_name_ranks_that_file_first() {
    both_caches(|engine, _, root| {
        for q in ["father.md", "father", "FATHER.MD"] {
            let r = search(&engine, root, q, SearchMode::Hybrid);
            assert_eq!(top(&r), "father.md", "query {q:?}: {:#?}", r.hits);
            assert!(r.hits[0].signals.filename);
        }
    });
}

#[test]
fn identifiers_are_found_by_keyword() {
    both_caches(|engine, _, root| {
        let r = search(&engine, root, "INC-18372", SearchMode::Hybrid);
        assert_eq!(top(&r), "incident.md");
        assert!(r.hits[0].signals.lexical.is_some());
        // Only the incident note contains the phrase "inc 18372".
        let lexical = search(&engine, root, "INC-18372", SearchMode::Lexical);
        assert_eq!(lexical.hits.len(), 1);
    });
}

#[test]
fn lexical_mode_never_embeds_the_query() {
    both_caches(|engine, counter, root| {
        let r = search(&engine, root, "basil tomatoes", SearchMode::Lexical);
        assert_eq!(top(&r), "garden.md");
        assert_eq!(counter.queries.load(Ordering::SeqCst), 0);
        search(&engine, root, "basil tomatoes", SearchMode::Hybrid);
        assert_eq!(counter.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn semantic_mode_scores_are_cosines() {
    both_caches(|engine, _, root| {
        let r = search(&engine, root, "basil tomatoes", SearchMode::Semantic);
        for h in &r.hits {
            assert_eq!(Some(h.score), h.signals.semantic);
            assert!(h.signals.lexical.is_none() && !h.signals.filename);
        }
    });
}

#[test]
fn edits_and_deletions_update_the_keyword_index() {
    let temp = corpus();
    let db = TempDir::new().unwrap();
    let (engine, _) = engine(SqliteCache::open(db.path().join("cache.sqlite3")).unwrap());
    let root = temp.path();
    assert_eq!(
        search(&engine, root, "basil", SearchMode::Lexical)
            .hits
            .len(),
        1
    );

    fs::write(root.join("garden.md"), "Only roses this year.").unwrap();
    assert!(
        search(&engine, root, "basil", SearchMode::Lexical)
            .hits
            .is_empty()
    );
    assert_eq!(
        search(&engine, root, "roses", SearchMode::Lexical)
            .hits
            .len(),
        1
    );

    fs::remove_file(root.join("garden.md")).unwrap();
    engine.index().paths([root]).execute().unwrap();
    assert!(
        search(&engine, root, "roses", SearchMode::Lexical)
            .hits
            .is_empty()
    );
}

#[test]
fn upgrading_a_v1_cache_keeps_every_embedding() {
    let temp = corpus();
    let db = TempDir::new().unwrap();
    let file = db.path().join("cache.sqlite3");
    {
        let (engine, _) = engine(SqliteCache::open(&file).unwrap());
        engine.index().paths([temp.path()]).execute().unwrap();
    }
    // Pretend the file was written by schema v1, before the keyword index.
    let conn = rusqlite::Connection::open(&file).unwrap();
    conn.execute_batch(
        "UPDATE meta SET value = '1' WHERE key = 'schema_version'; DELETE FROM segments_fts;",
    )
    .unwrap();
    drop(conn);

    let (engine, _) = engine(SqliteCache::open(&file).unwrap());
    let r = search(&engine, temp.path(), "INC-18372", SearchMode::Hybrid);
    assert_eq!(
        r.stats.sync.segments_embedded, 0,
        "no vector was recomputed"
    );
    assert_eq!(
        r.stats.sync.resources_updated, 4,
        "every file was re-indexed"
    );
    assert_eq!(top(&r), "incident.md");
}
