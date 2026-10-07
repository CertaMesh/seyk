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
        // Only the incident note contains both identifier tokens in one field.
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
fn identifiers_match_without_order_or_adjacency_in_both_caches() {
    both_caches(|engine, _, root| {
        fs::write(root.join("incident.md"), "18372 intervening words INC").unwrap();
        fs::write(root.join("garden.md"), "restore unrelated words Session").unwrap();
        // Partial matches in different fields must not combine into a group.
        fs::write(root.join("session.md"), "restore only").unwrap();
        for (query, name) in [
            ("INC-18372", "incident.md"),
            ("Session::restore()", "garden.md"),
        ] {
            let r = search(&engine, root, query, SearchMode::Lexical);
            assert_eq!(r.hits.len(), 1);
            assert_eq!(top(&r), name);
        }
    });
}

#[test]
fn persisted_positions_are_sorted_in_every_field() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("zeta alpha zeta.md");
    fs::write(
        &file,
        "# Zebra Alpha Zebra\n\nZürich Café κόσμος नमस्ते 東京 aⅣ2 Café",
    )
    .unwrap();
    let db = TempDir::new().unwrap();
    let cache_file = db.path().join("cache.sqlite3");
    let (engine, _) = engine(SqliteCache::open(&cache_file).unwrap());
    engine.index().paths([&file]).execute().unwrap();
    let conn = rusqlite::Connection::open(cache_file).unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, segments_fts, instance);",
    )
    .unwrap();
    for field in ["name", "path", "heading", "body"] {
        let mut stmt = conn
            .prepare("SELECT doc, term FROM temp.vocab WHERE col=?1 ORDER BY doc, offset")
            .unwrap();
        let tokens = stmt
            .query_map([field], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(!tokens.is_empty(), "{field}");
        assert!(
            tokens
                .windows(2)
                .all(|pair| pair[0].0 != pair[1].0 || pair[0].1 <= pair[1].1),
            "{field}: {tokens:?}"
        );
        if field == "name" {
            assert_eq!(tokens.iter().filter(|(_, term)| term == "zeta").count(), 2);
        }
    }
}

#[test]
fn upgrading_v1_keeps_embeddings_across_partial_root_rebuilds() {
    check_partial_root_migration("1");
}

#[test]
fn upgrading_v2_keeps_embeddings_across_partial_root_rebuilds() {
    check_partial_root_migration("2");
}

/// Whether a token survives in the FTS index's own storage. A plain DELETE on a
/// contentless table only tombstones rows, so queries miss the token while its
/// bytes stay in `segments_fts_data`.
fn fts_data_contains(conn: &rusqlite::Connection, token: &str) -> bool {
    let mut stmt = conn.prepare("SELECT block FROM segments_fts_data").unwrap();
    let blocks = stmt
        .query_map([], |r| r.get::<_, Vec<u8>>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    blocks
        .iter()
        .any(|b| b.windows(token.len()).any(|w| w == token.as_bytes()))
}

#[test]
fn clearing_the_cache_leaves_no_tokens_in_the_keyword_index() {
    let root = TempDir::new().unwrap();
    fs::write(root.path().join("note.md"), "Quetzalcoatl visited.").unwrap();
    let db = TempDir::new().unwrap();
    let file = db.path().join("cache.sqlite3");
    let (engine, _) = engine(SqliteCache::open(&file).unwrap());
    engine.index().paths([root.path()]).execute().unwrap();
    let conn = rusqlite::Connection::open(&file).unwrap();
    assert!(fts_data_contains(&conn, "quetzalcoatl"));
    engine.with_cache(|c| c.clear()).unwrap();
    assert!(!fts_data_contains(&conn, "quetzalcoatl"));
}

fn check_partial_root_migration(version: &str) {
    let root_a = TempDir::new().unwrap();
    let root_b = TempDir::new().unwrap();
    fs::write(root_a.path().join("garden.md"), "Tomatoes need basil.").unwrap();
    fs::write(
        root_b.path().join("incident.md"),
        "Outage review for INC-18372.",
    )
    .unwrap();
    let db = TempDir::new().unwrap();
    let file = db.path().join("cache.sqlite3");
    {
        let (engine, _) = engine(SqliteCache::open(&file).unwrap());
        let stats = engine
            .index()
            .paths([root_a.path(), root_b.path()])
            .execute()
            .unwrap();
        assert_eq!(stats.sync.segments_embedded, 2);
    }
    let conn = rusqlite::Connection::open(&file).unwrap();
    conn.execute(
        "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
        [version],
    )
    .unwrap();
    if version == "1" {
        // v1 has the same resource/segment columns as v3, but no FTS table.
        conn.execute_batch("DROP TABLE segments_fts;").unwrap();
    } else {
        // Include an orphan row to prove migration clears the entire old index.
        conn.execute(
            "INSERT INTO segments_fts(rowid,body) VALUES(99999,'obsolete ordered words')",
            [],
        )
        .unwrap();
    }
    let vectors = || {
        conn.prepare("SELECT input_hash,vector FROM embeddings ORDER BY input_hash")
            .unwrap()
            .query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    let count = |sql: &str| conn.query_row(sql, [], |r| r.get::<_, i64>(0)).unwrap();
    let before = vectors();
    // Close every cache/engine between phases, as when separate processes sync roots.
    drop(SqliteCache::open(&file).unwrap());
    assert_eq!(count("SELECT count(*) FROM resources"), 2);
    assert_eq!(count("SELECT count(*) FROM segments"), 2);
    assert_eq!(count("SELECT count(*) FROM segments_fts"), 0);
    assert!(!fts_data_contains(&conn, "obsolete"));
    assert_eq!(
        count("SELECT count(*) FROM resources WHERE modified_ns IS NULL AND content_hash = ''"),
        2
    );
    assert_eq!(vectors(), before);
    assert_eq!(
        conn.query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "3"
    );

    let assert_unsynced_search = |engine: &SearchEngine| {
        let semantic = engine
            .search("INC-18372")
            .mode(SearchMode::Semantic)
            .paths([root_b.path()])
            .sync(false)
            .execute()
            .unwrap();
        assert_eq!(semantic.hits.len(), 1);
        assert_eq!(top(&semantic), "incident.md");
        assert!(semantic.hits[0].signals.semantic.is_some());
        let lexical = engine
            .search("INC-18372")
            .mode(SearchMode::Lexical)
            .paths([root_b.path()])
            .sync(false)
            .execute()
            .unwrap();
        assert!(
            lexical.hits.is_empty(),
            "keywords stay empty until this root is rebuilt"
        );
    };
    {
        let (engine, counter) = engine(SqliteCache::open(&file).unwrap());
        assert_unsynced_search(&engine);
        let stats = engine.index().paths([root_a.path()]).execute().unwrap();
        assert_eq!(stats.sync.resources_updated, 1);
        assert_eq!(stats.sync.segments_embedded, 0);
        assert_eq!(counter.inner.resource_count(), 0);
        assert_eq!(
            vectors(),
            before,
            "root B's vectors must survive root A's orphan pruning"
        );
        assert_eq!(
            count("SELECT count(*) FROM resources WHERE modified_ns IS NULL AND content_hash = ''"),
            1
        );
    }
    {
        let (engine, counter) = engine(SqliteCache::open(&file).unwrap());
        assert_unsynced_search(&engine);
        let r = search(&engine, root_b.path(), "INC-18372", SearchMode::Lexical);
        assert_eq!(r.stats.sync.resources_updated, 1);
        assert_eq!(
            r.stats.sync.segments_embedded, 0,
            "no vector was recomputed"
        );
        assert_eq!(counter.inner.resource_count(), 0);
        assert_eq!(top(&r), "incident.md");
        assert_eq!(vectors(), before);
        assert_eq!(
            count("SELECT count(*) FROM resources WHERE content_hash = ''"),
            0
        );
        assert_eq!(
            count("SELECT count(*) FROM segments_fts WHERE segments_fts MATCH 'obsolete'"),
            0
        );
    }
}
