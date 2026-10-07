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
        .to_file_path()
        .expect("file resource")
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
        "4"
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

#[test]
fn equal_score_candidates_are_stable_before_fusion_and_top_k() {
    let root = TempDir::new().unwrap();
    // More than the hybrid candidate depth (50), to exercise its cutoff too.
    for i in (0..60).rev() {
        fs::write(root.path().join(format!("note-{i:02}.md")), "quartz pebble").unwrap();
    }
    let check = |engine: SearchEngine, counter: Arc<Counting>| {
        engine.index().paths([root.path()]).execute().unwrap();
        let vector = counter.embed_query("quartz").unwrap();
        engine
            .with_cache(|cache| {
                let resources = cache.resource_index(engine.profile_id())?;
                for path in resources.keys() {
                    let resource = cache.resource(engine.profile_id(), path)?.unwrap();
                    for segment in resource.segments {
                        cache.put_embeddings(
                            engine.profile_id(),
                            &[(segment.input_hash, vector.as_slice().to_vec())],
                        )?;
                    }
                }
                Ok(())
            })
            .unwrap();
        let mut orders = Vec::new();
        for mode in [
            SearchMode::Lexical,
            SearchMode::Hybrid,
            SearchMode::Semantic,
        ] {
            let result = engine
                .search("quartz")
                .paths([root.path()])
                .mode(mode)
                .sync(false)
                .limit(3)
                .execute()
                .unwrap();
            let order: Vec<_> = result
                .hits
                .iter()
                .map(|h| {
                    h.resource
                        .to_file_path()
                        .expect("file resource")
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            assert_eq!(order, ["note-00.md", "note-01.md", "note-02.md"]);
            orders.push(order);
        }
        orders
    };
    let db = TempDir::new().unwrap();
    let (e, c) = engine(SqliteCache::open(db.path().join("cache.sqlite3")).unwrap());
    let expected = check(e, c);
    for _ in 0..20 {
        let (e, c) = engine(MemoryCache::new());
        assert_eq!(check(e, c), expected);
    }
}

#[test]
fn invalid_weights_fail_before_indexing_or_embedding_in_every_mode() {
    both_caches(|engine, counter, root| {
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Lexical,
            SearchMode::Semantic,
        ] {
            for weight in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let result = engine
                    .search("basil")
                    .paths([root])
                    .mode(mode)
                    .lexical_weight(weight)
                    .execute();
                assert!(
                    matches!(result, Err(seyk::Error::Config(_))),
                    "{mode:?}, {weight}: {result:?}"
                );
            }
        }
        assert_eq!(counter.queries.load(Ordering::SeqCst), 0);
        assert_eq!(counter.inner.resource_count(), 0);
    });
}

#[test]
fn zero_hybrid_weight_disables_only_keywords_and_keeps_scores_finite() {
    both_caches(|engine, _, root| {
        // The cosine threshold suppresses semantics, leaving only keywords.
        let result = engine
            .search("basil")
            .paths([root])
            .min_score(2.0)
            .lexical_weight(0.0)
            .execute()
            .unwrap();
        assert!(result.hits.is_empty());
        let named = engine
            .search("father.md")
            .paths([root])
            .min_score(2.0)
            .lexical_weight(0.0)
            .execute()
            .unwrap();
        assert_eq!(named.hits.len(), 1);
        assert_eq!(named.hits[0].score, 1.0);
        assert!(named.hits[0].signals.filename);
        assert!(named.hits[0].signals.lexical.is_none());
        let semantic = engine
            .search("basil")
            .paths([root])
            .lexical_weight(0.0)
            .execute()
            .unwrap();
        assert!(!semantic.hits.is_empty());
        assert!(
            semantic
                .hits
                .iter()
                .all(|h| h.score.is_finite() && h.signals.lexical.is_none())
        );
        for weight in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let result = engine
                .search("basil")
                .paths([root])
                .min_score(2.0)
                .lexical_weight(weight)
                .execute()
                .unwrap();
            assert!(
                result
                    .hits
                    .iter()
                    .all(|h| h.score.is_finite() && h.score > 0.0 && h.score <= 1.0)
            );
        }
        let missing = engine
            .search("unfindabletoken")
            .paths([root])
            .min_score(2.0)
            .execute()
            .unwrap();
        assert!(missing.hits.is_empty());
    });
}

#[test]
fn valid_weight_overrides_do_not_change_lexical_or_semantic_mode() {
    both_caches(|engine, _, root| {
        for mode in [SearchMode::Lexical, SearchMode::Semantic] {
            let baseline = search(&engine, root, "father", mode);
            for weight in [0.0, 10.0, f32::MAX] {
                let result = engine
                    .search("father")
                    .paths([root])
                    .mode(mode)
                    .lexical_weight(weight)
                    .execute()
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&result.hits).unwrap(),
                    serde_json::to_value(&baseline.hits).unwrap()
                );
            }
        }
    });
}

#[test]
fn spaced_names_and_cross_platform_suffixes_keep_the_filename_signal() {
    both_caches(|engine, _, root| {
        fs::create_dir_all(root.join("my people")).unwrap();
        fs::write(
            root.join("my people/My Notes.md"),
            "Synthetic notebook entry.",
        )
        .unwrap();
        for query in [
            "my notes.md",
            "my notes",
            " \"My Notes.md\" ",
            "`my notes`",
            "my people/My Notes.md",
            r"my people\My Notes.md",
        ] {
            let result = search(&engine, root, query, SearchMode::Hybrid);
            assert_eq!(top(&result), "My Notes.md", "{query}");
            assert!(result.hits[0].signals.filename, "{query}");
        }
        let result = search(&engine, root, "my father", SearchMode::Hybrid);
        assert!(result.hits.iter().all(|h| !h.signals.filename));
    });
}
