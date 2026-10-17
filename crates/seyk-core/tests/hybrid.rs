mod common;
#[path = "support/golden_provider.rs"]
mod golden_provider;
#[cfg(feature = "pdf")]
#[path = "support/scan_fixture.rs"]
mod hydration_fixture;

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::FakeEmbedder;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingCache, MemoryCache, Modality, ModelIdentity,
    ResourceInput, SearchEngine, SearchMode, SearchResults, SqliteCache,
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
    fn embed_query(&self, q: &str) -> seyk_embed::Result<Embedding> {
        self.queries.fetch_add(1, Ordering::SeqCst);
        self.inner.embed_query(q)
    }
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        self.inner.embed_resources(inputs)
    }

    fn embed_batch(
        &self,
        requests: &[seyk_embed::EmbeddingRequest<'_>],
    ) -> seyk_embed::Result<Vec<Embedding>> {
        if requests
            .iter()
            .all(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
        {
            let inputs: Vec<_> = requests.iter().map(|r| r.input.clone()).collect();
            return self.embed_resources(&inputs);
        }
        requests
            .iter()
            .map(|r| match &r.input {
                ResourceInput::Text { text, .. }
                    if r.task == seyk_embed::EmbeddingTask::SearchQuery =>
                {
                    self.embed_query(text)
                }
                _ => self
                    .embed_resources(std::slice::from_ref(&r.input))
                    .map(|mut v| v.remove(0)),
            })
            .collect()
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

fn both_caches(mut test: impl FnMut(SearchEngine, Arc<Counting>, &Path)) {
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
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, {}, instance);",
        common::legacy::table(engine.profile_id())
    ))
    .unwrap();
    for field in ["name", "path", "heading", "body"] {
        let table = common::legacy::table(engine.profile_id());
        let stored = conn
            .prepare(&format!("SELECT {field} FROM {table}"))
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        for text in stored {
            assert_eq!(text, seyk_core::lexical::order_free_text(&text));
        }
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
    let tables = conn.prepare("SELECT name FROM sqlite_master WHERE name='segments_fts_data' OR name GLOB 'segments_fts_p_*_data'").unwrap()
        .query_map([], |r| r.get::<_, String>(0)).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
    tables.iter().any(|table| {
        conn.prepare(&format!("SELECT block FROM {table}"))
            .unwrap()
            .query_map([], |r| r.get::<_, Vec<u8>>(0))
            .unwrap()
            .any(|b| {
                b.unwrap()
                    .windows(token.len())
                    .any(|w| w == token.as_bytes())
            })
    })
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
    common::legacy::downgrade(&conn, version);
    if version == "2" {
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
    let profile: String = conn
        .query_row("SELECT id FROM profiles", [], |r| r.get(0))
        .unwrap();
    let table = common::legacy::table(&profile);
    assert_eq!(count(&format!("SELECT count(*) FROM {table}")), 0);
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
        "6"
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
            count(&format!(
                "SELECT count(*) FROM {table} WHERE {table} MATCH 'obsolete'"
            )),
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
                    matches!(result, Err(seyk_core::Error::Config(_))),
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

/// A transparent cache wrapper with read accounting and controlled readback gaps.
mod hydration {
    use super::*;
    use seyk_core::cache::{
        CacheProfile, CacheStats, CachedResource, CachedResourceState, LexicalHit, ResourceWrite,
        StoredVector,
    };
    use seyk_core::lexical::LexicalQuery;
    use seyk_core::{ResourceId, ResourceMetadata, ResourceVersion, Result};
    use std::collections::{HashMap, HashSet};
    use std::sync::{Mutex, atomic::AtomicBool};

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum Readback {
        #[default]
        Normal,
        MissingResource,
        MissingOrdinals,
        ErrorOnce,
        ReplacementEvidence,
    }

    #[derive(Default)]
    struct Reads {
        count: AtomicUsize,
        readback: Mutex<Readback>,
        partial_vectors: AtomicBool,
        term_batches: Mutex<Vec<Vec<(ResourceId, usize)>>>,
    }
    impl Reads {
        fn next_readback(&self) -> Readback {
            let mut state = self.readback.lock().unwrap();
            let current = *state;
            if current == Readback::ErrorOnce {
                *state = Readback::Normal;
            }
            current
        }
    }
    struct CountCache {
        inner: Box<dyn EmbeddingCache>,
        reads: Arc<Reads>,
    }
    impl EmbeddingCache for CountCache {
        fn register_profile(&mut self, p: &str, v: &CacheProfile) -> Result<()> {
            self.inner.register_profile(p, v)
        }
        fn resource_index(&self, p: &str) -> Result<HashMap<ResourceId, CachedResourceState>> {
            self.inner.resource_index(p)
        }
        fn resource(&self, p: &str, r: &ResourceId) -> Result<Option<CachedResource>> {
            self.reads.count.fetch_add(1, Ordering::SeqCst);
            match self.reads.next_readback() {
                Readback::MissingResource => Ok(None),
                Readback::MissingOrdinals => Ok(self.inner.resource(p, r)?.map(|mut r| {
                    r.segments.truncate(1);
                    r
                })),
                Readback::ErrorOnce => Err(seyk_core::Error::Config(
                    "injected resource read failure".into(),
                )),
                Readback::ReplacementEvidence => Ok(self.inner.resource(p, r)?.map(|mut r| {
                    for segment in &mut r.segments {
                        segment.locator =
                            if r.metadata.media_type.as_deref() == Some("application/pdf") {
                                seyk_core::ResourceLocator::Page { page: 99 }
                            } else {
                                seyk_core::ResourceLocator::Text {
                                    byte_start: 999_000,
                                    byte_end: 999_010,
                                    line_start: None,
                                    line_end: None,
                                }
                            };
                        segment.heading = Some("replacement evidence".into());
                    }
                    r
                })),
                Readback::Normal => self.inner.resource(p, r),
            }
        }
        fn put_resource(&mut self, p: &str, r: &ResourceId, v: &ResourceWrite<'_>) -> Result<()> {
            self.inner.put_resource(p, r, v)
        }
        fn touch_resource(
            &mut self,
            p: &str,
            r: &ResourceId,
            m: &ResourceMetadata,
            v: &ResourceVersion,
        ) -> Result<()> {
            self.inner.touch_resource(p, r, m, v)
        }
        fn remove_resources(&mut self, p: &str, r: &[ResourceId]) -> Result<()> {
            self.inner.remove_resources(p, r)
        }
        fn known_inputs(&self, p: &str, h: &[String]) -> Result<HashSet<String>> {
            self.inner.known_inputs(p, h)
        }
        fn put_embeddings(&mut self, p: &str, v: &[(String, Vec<f32>)]) -> Result<()> {
            self.inner.put_embeddings(p, v)
        }
        fn vectors_for(&self, p: &str, r: &HashSet<ResourceId>) -> Result<Vec<StoredVector>> {
            let mut rows = self.inner.vectors_for(p, r)?;
            if self.reads.partial_vectors.load(Ordering::SeqCst) {
                rows.retain(|v| v.ordinal == 0);
            }
            Ok(rows)
        }
        fn lexical_search(
            &self,
            p: &str,
            q: &LexicalQuery,
            r: &HashSet<ResourceId>,
            n: usize,
        ) -> Result<Vec<LexicalHit>> {
            self.inner.lexical_search(p, q, r, n)
        }
        fn lexical_matched_terms(
            &self,
            p: &str,
            q: &LexicalQuery,
            keys: &[(ResourceId, usize)],
        ) -> Result<Vec<Vec<String>>> {
            self.reads.term_batches.lock().unwrap().push(keys.to_vec());
            self.inner.lexical_matched_terms(p, q, keys)
        }
        fn prune_orphans(&mut self, p: &str) -> Result<u64> {
            self.inner.prune_orphans(p)
        }
        fn stats(&self) -> Result<CacheStats> {
            self.inner.stats()
        }
        fn clear(&mut self) -> Result<()> {
            self.inner.clear()
        }
    }

    // Independently selectable capabilities, deliberately with the same identity:
    // cached evidence must remain safe under sync(false) capability changes.
    struct Capabilities {
        inner: FakeEmbedder,
        text: bool,
        image: bool,
    }
    impl Embedder for Capabilities {
        fn identity(&self) -> &ModelIdentity {
            self.inner.identity()
        }
        fn supports(&self, m: Modality) -> bool {
            (m == Modality::Text && self.text) || (m == Modality::Image && self.image)
        }
        fn embed_batch(
            &self,
            r: &[seyk_embed::EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<Embedding>> {
            r.iter()
                .map(|_| Embedding::from_raw(vec![1.0; 256], Dimension::D256))
                .collect()
        }
    }

    fn counted(sqlite: bool, db: &Path, text: bool, image: bool) -> (SearchEngine, Arc<Reads>) {
        let reads = Arc::new(Reads::default());
        let inner: Box<dyn EmbeddingCache> = if sqlite {
            Box::new(SqliteCache::open(db).unwrap())
        } else {
            Box::new(MemoryCache::new())
        };
        let engine = SearchEngine::builder()
            .embedder(Capabilities {
                inner: FakeEmbedder::new(Dimension::D256, "hydration"),
                text,
                image,
            })
            .cache(CountCache {
                inner,
                reads: reads.clone(),
            })
            .build()
            .unwrap();
        (engine, reads)
    }

    fn query(
        e: &SearchEngine,
        root: &Path,
        q: &str,
        mode: SearchMode,
    ) -> seyk_core::Result<SearchResults> {
        e.search(q)
            .paths([root])
            .mode(mode)
            .limit(100)
            .previews(None)
            .sync(false)
            .execute()
    }

    #[test]
    fn terms_are_requested_only_for_final_contributing_hits() {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.md");
            fs::write(
                &path,
                (0..80)
                    .map(|i| format!("# Section {i}\n\nbasil garden {i}\n\n"))
                    .collect::<String>(),
            )
            .unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            for (mode, weight, query) in [
                (SearchMode::Hybrid, 1.0, "basil"),
                (SearchMode::Lexical, 1.0, "basil"),
                (SearchMode::Hybrid, 0.0, "basil"),
                (SearchMode::Semantic, 1.0, "basil"),
                (SearchMode::Hybrid, 1.0, "absent"),
            ] {
                reads.term_batches.lock().unwrap().clear();
                let result = e
                    .search(query)
                    .paths([&path])
                    .mode(mode)
                    .lexical_weight(weight)
                    .limit(2)
                    .sync(false)
                    .previews(None)
                    .execute()
                    .unwrap();
                let batches = reads.term_batches.lock().unwrap();
                let expected = result
                    .hits
                    .iter()
                    .filter(|h| h.signals.lexical.is_some())
                    .count();
                if expected == 0 {
                    assert!(batches.is_empty());
                } else {
                    assert_eq!(batches.len(), 1);
                    assert_eq!(batches[0].len(), expected);
                    for (resource, ordinal) in &batches[0] {
                        let cached = e
                            .with_cache(|c| c.resource(e.profile_id(), resource))
                            .unwrap()
                            .unwrap();
                        assert!(result.hits.iter().any(|h| h.resource == *resource
                            && h.locator == cached.segments[*ordinal].locator));
                    }
                }
            }
            // Hits discarded during final hydration never request term metadata.
            reads.term_batches.lock().unwrap().clear();
            *reads.readback.lock().unwrap() = Readback::MissingResource;
            assert!(
                e.search("basil")
                    .paths([&path])
                    .mode(SearchMode::Lexical)
                    .sync(false)
                    .execute()
                    .unwrap()
                    .hits
                    .is_empty()
            );
            assert!(reads.term_batches.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn lexical_and_filename_hydrate_once_and_partial_vectors_are_not_complete() {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.md");
            fs::write(
                &path,
                (0..8)
                    .map(|i| format!("# Section {i}\n\nbasil garden number {i}\n\n"))
                    .collect::<String>(),
            )
            .unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            reads.partial_vectors.store(true, Ordering::SeqCst);
            for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
                for q in ["basil", "garden.md"] {
                    reads.count.store(0, Ordering::SeqCst);
                    let result = query(&e, &path, q, mode).unwrap();
                    assert!(result.hits.len() >= 8, "{mode:?}: {result:?}");
                    assert_eq!(reads.count.load(Ordering::SeqCst), 1, "{mode:?} {q}");
                }
            }
            // A new request must reload; completion is never engine-global.
            reads.count.store(0, Ordering::SeqCst);
            query(&e, &path, "basil", SearchMode::Lexical).unwrap();
            assert_eq!(reads.count.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn complete_misses_are_reused_and_read_errors_propagate_without_poisoning_later_searches() {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.md");
            fs::write(
                &path,
                "# One\n\nbasil\n\n# Two\n\nbasil\n\n# Three\n\nbasil",
            )
            .unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            for (readback, expected) in [
                (Readback::MissingResource, 0),
                (Readback::MissingOrdinals, 1),
            ] {
                *reads.readback.lock().unwrap() = readback;
                reads.count.store(0, Ordering::SeqCst);
                let result = query(&e, &path, "basil", SearchMode::Lexical).unwrap();
                assert_eq!(result.hits.len(), expected);
                assert_eq!(reads.count.load(Ordering::SeqCst), 1);
            }
            *reads.readback.lock().unwrap() = Readback::ErrorOnce;
            assert!(
                query(&e, &path, "basil", SearchMode::Lexical)
                    .unwrap_err()
                    .to_string()
                    .contains("injected")
            );
            assert!(
                query(&e, &path, "basil", SearchMode::Lexical)
                    .unwrap()
                    .hits
                    .len()
                    >= 3
            );
        }
    }

    fn preserves_scored_rows_after_complete_lookup(readback: Readback) {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.md");
            fs::write(
                &path,
                "# One\n\nbasil\n\n# Two\n\nbasil\n\n# Three\n\nbasil",
            )
            .unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            let before = query(&e, &path, "garden.md", SearchMode::Hybrid).unwrap();
            assert_eq!(before.hits.len(), 3);
            // All vector rows remain available, while the later resource read
            // observes a concurrent indexer's removal, truncation or replacement.
            *reads.readback.lock().unwrap() = readback;
            reads.count.store(0, Ordering::SeqCst);
            let after = query(&e, &path, "garden.md", SearchMode::Hybrid).unwrap();
            assert_eq!(
                after.hits.len(),
                3,
                "sqlite={sqlite}, readback={readback:?}"
            );
            assert_eq!(reads.count.load(Ordering::SeqCst), 1);
            assert_eq!(
                after.hits.iter().filter(|h| h.signals.filename).count(),
                usize::from(readback != Readback::MissingResource)
            );
            for (old, new) in before.hits.iter().zip(&after.hits) {
                assert_eq!(new.resource, old.resource);
                assert_eq!(new.locator, old.locator);
                assert_eq!(new.heading, old.heading);
                assert_eq!(new.metadata, old.metadata);
                assert_eq!(new.signals.semantic, old.signals.semantic);
            }
        }
    }

    #[test]
    fn hybrid_keeps_scored_rows_after_a_missing_resource_lookup() {
        preserves_scored_rows_after_complete_lookup(Readback::MissingResource);
    }

    #[test]
    fn hybrid_keeps_scored_rows_after_a_truncated_resource_lookup() {
        preserves_scored_rows_after_complete_lookup(Readback::MissingOrdinals);
    }

    #[test]
    fn hybrid_keeps_scored_rows_after_a_replacement_locator_lookup() {
        preserves_scored_rows_after_complete_lookup(Readback::ReplacementEvidence);
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn hybrid_complete_lookup_eligibility_does_not_use_preserved_vector_rows() {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.pdf");
            let text = b"BT /F1 12 Tf 50 720 Td (Public basil garden.) Tj ET";
            fs::write(&path, fixture::document(&[(text.as_slice(), ""); 3], None)).unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            let before = query(&e, &path, "garden.pdf", SearchMode::Hybrid).unwrap();
            assert_eq!(before.hits.len(), 3);
            *reads.readback.lock().unwrap() = Readback::ReplacementEvidence;
            reads.count.store(0, Ordering::SeqCst);
            let after = query(&e, &path, "garden.pdf", SearchMode::Hybrid).unwrap();
            assert_eq!(after.hits.len(), 3);
            assert_eq!(reads.count.load(Ordering::SeqCst), 1);
            for (old, new) in before.hits.iter().zip(&after.hits) {
                assert_eq!(new.locator, old.locator);
                assert_eq!(new.heading, old.heading);
                assert_eq!(new.signals.semantic, old.signals.semantic);
                // The complete lookup contains only unsupported image pages:
                // vector rows remain semantic evidence, not filename/lexical candidates.
                assert!(!new.signals.filename);
                assert!(new.signals.lexical.is_none());
            }
        }
    }

    #[cfg(feature = "pdf")]
    use super::hydration_fixture as fixture;

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_prepass_filename_and_hydration_share_complete_reads() {
        for sqlite in [false, true] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("garden.pdf");
            fs::write(&path, fixture::text(&["basil garden"])).unwrap();
            let (e, reads) = counted(sqlite, &dir.path().join("cache.db"), true, false);
            e.index().paths([&path]).execute().unwrap();
            for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
                reads.count.store(0, Ordering::SeqCst);
                assert_eq!(query(&e, &path, "garden.pdf", mode).unwrap().hits.len(), 1);
                assert_eq!(reads.count.load(Ordering::SeqCst), 1);
            }
        }
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_prepass_requires_both_feature_gates_and_both_capabilities() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("garden.pdf");
        let db = dir.path().join("cache.db");
        fs::write(&path, fixture::text(&["basil garden"])).unwrap();
        let (e, _) = counted(true, &db, true, true);
        e.index().paths([&path]).execute().unwrap();
        drop(e);
        for (text, image) in [(true, true), (false, true), (true, false), (false, false)] {
            let (e, reads) = counted(true, &db, text, image);
            for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
                reads.count.store(0, Ordering::SeqCst);
                query(&e, &path, "absent", mode).unwrap();
                let fast = cfg!(feature = "pdf-render") && text && image;
                assert_eq!(
                    reads.count.load(Ordering::SeqCst),
                    usize::from(!fast),
                    "text={text} image={image} {mode:?}"
                );
            }
        }
    }

    /// Stress the sibling-retention tradeoff separately from the PDF prepass.
    #[test]
    #[ignore = "synthetic 1000-segment hydration measurement"]
    fn measure_large_resource_hydration() {
        use std::time::Instant;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("garden.md");
        let source = (0..1000)
            .map(|i| format!("# Section {i}\n\nbasil garden number {i}\n\n"))
            .collect::<String>();
        fs::write(&path, source).unwrap();
        let (e, reads) = counted(true, &dir.path().join("cache.db"), true, false);
        e.index().paths([&path]).execute().unwrap();
        let resource = seyk_core::ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
        let count = e
            .with_cache(|c| c.resource(e.profile_id(), &resource))
            .unwrap()
            .unwrap()
            .segments
            .len();
        assert_eq!(count, 1000);
        query(&e, &path, "basil", SearchMode::Lexical).unwrap();
        reads.count.store(0, Ordering::SeqCst);
        let mut samples = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            let result = query(&e, &path, "basil", SearchMode::Lexical).unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(result.hits.len(), 100);
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "P7_LARGE segments={count} hits=100 sqlite reads_per_search={} median_ms={:.3} p95_ms={:.3}",
            reads.count.load(Ordering::SeqCst) / 20,
            samples[10],
            samples[18]
        );
    }

    /// Public synthetic fixture only. Run explicitly with --ignored --nocapture.
    #[cfg(feature = "pdf-render")]
    #[test]
    #[ignore = "synthetic 1000-PDF hydration measurement"]
    fn measure_pdf_hydration() {
        use std::time::Instant;
        let dir = TempDir::new().unwrap();
        let corpus = dir.path().join("pdfs");
        fs::create_dir(&corpus).unwrap();
        let bytes = fixture::text(&["Public basil garden with lavender."]);
        for i in 0..1000 {
            fs::write(corpus.join(format!("garden-{i:04}.pdf")), &bytes).unwrap();
        }
        let (e, reads) = counted(true, &dir.path().join("cache.db"), true, true);
        e.index().paths([&corpus]).execute().unwrap();
        for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
            for q in ["absent", "basil", "garden-0000.pdf"] {
                query(&e, &corpus, q, mode).unwrap(); // warm up
                let mut samples = Vec::new();
                reads.count.store(0, Ordering::SeqCst);
                for _ in 0..20 {
                    let start = Instant::now();
                    let r = query(&e, &corpus, q, mode).unwrap();
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                    std::hint::black_box(r);
                }
                samples.sort_by(f64::total_cmp);
                eprintln!(
                    "P7_MEASURE pdfs=1000 pages=1 sqlite mode={mode:?} query={q} reads_per_search={} median_ms={:.3} p95_ms={:.3}",
                    reads.count.load(Ordering::SeqCst) / 20,
                    samples[10],
                    samples[18]
                );
            }
        }
        // TempDir removes the generated PDFs and SQLite cache on return.
    }
}

#[test]
fn lexical_fallback_finds_unembedded_files_without_touching_semantic_cache() {
    let root = corpus();
    let (engine, counter) = engine(MemoryCache::new());
    let before = engine.with_cache(|c| c.stats()).unwrap();
    let results = engine
        .search("INC-18372")
        .query("father.md")
        .paths([root.path()])
        .sync(false)
        .lexical_fallback(true)
        .execute()
        .unwrap();
    assert_eq!(
        results.degraded.unwrap().reason,
        seyk_core::DegradedReason::NotYetEmbedded
    );
    assert!(results.hits.iter().any(|h| h.signals.lexical.is_some()));
    assert!(results.hits.iter().any(|h| h.signals.filename));
    assert!(results.hits.iter().all(|h| h.signals.semantic.is_none()));
    assert_eq!(results.stats.sync.segments_embedded, 0);
    assert_eq!(counter.queries.load(Ordering::SeqCst), 0);
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(engine.with_cache(|c| c.stats()).unwrap()).unwrap()
    );
    for hit in &results.hits {
        engine.extract(hit).unwrap();
    }
}

#[test]
fn ready_results_match_with_lexical_fallback_opt_in() {
    let root = corpus();
    let (engine, _) = engine(MemoryCache::new());
    engine.index().paths([root.path()]).execute().unwrap();
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Lexical,
        SearchMode::Semantic,
    ] {
        let request = |fallback| {
            engine
                .search("father")
                .query("INC-18372")
                .paths([root.path()])
                .mode(mode)
                .sync(false)
                .lexical_fallback(fallback)
                .execute()
                .unwrap()
        };
        let mut baseline = request(false);
        let mut candidate = request(true);
        // Elapsed wall time is deliberately nondeterministic; every other byte,
        // including hits, scores, signals, locators, previews and stats, must match.
        baseline.stats.total_ms = 0;
        candidate.stats.total_ms = 0;
        assert!(candidate.degraded.is_none());
        assert_eq!(
            serde_json::to_vec(&baseline).unwrap(),
            serde_json::to_vec(&candidate).unwrap()
        );
    }
}

#[test]
fn model_free_lexical_search_refreshes_changes_and_honors_filters() {
    let root = corpus();
    let engine = SearchEngine::lexical().unwrap();
    let request = || {
        engine
            .search("INC-18372")
            .paths([root.path()])
            .sync(false)
            .execute()
            .unwrap()
    };
    let first = request();
    assert_eq!(
        first.degraded.unwrap().reason,
        seyk_core::DegradedReason::NoModel
    );
    assert_eq!(first.hits.len(), 1);
    fs::write(root.path().join("new.md"), "INC-18372 new evidence").unwrap();
    assert_eq!(request().hits.len(), 2);
    let filtered = engine
        .search("INC-18372")
        .paths([root.path()])
        .exclude("new.md")
        .execute()
        .unwrap();
    assert_eq!(filtered.hits.len(), 1);
    assert_eq!(
        engine.with_cache(|c| c.stats()).unwrap().profiles[0].embeddings,
        0
    );
}

#[cfg(feature = "pdf")]
#[test]
fn model_free_pdf_words_and_names_have_extractable_page_locators() {
    #[path = "support/pdf_fixture.rs"]
    mod fixture;
    let root = TempDir::new().unwrap();
    fs::write(
        root.path().join("quartz.pdf"),
        fixture::pdf(&["quartz lighthouse keeper"], fixture::Kind::Plain),
    )
    .unwrap();
    let engine = SearchEngine::lexical().unwrap();
    let results = engine
        .search("quartz")
        .paths([root.path()])
        .execute()
        .unwrap();
    assert_eq!(results.hits.len(), 1);
    let hit = &results.hits[0];
    assert!(hit.signals.lexical.is_some() && hit.signals.filename);
    assert!(matches!(
        hit.locator,
        seyk_core::ResourceLocator::PageText { page: 1, .. }
    ));
    assert!(engine.extract(hit).unwrap().text.contains("quartz"));
    assert_eq!(results.stats.sync.segments_embedded, 0);
}

#[test]
fn fallback_search_in_embedding_callback_does_not_wait_for_indexing_lease() {
    let root = corpus();
    let (engine, _) = engine(MemoryCache::new());
    let checked = AtomicUsize::new(0);
    let progress = |event| {
        if matches!(event, seyk_core::ProgressEvent::Embedding { done: 0, total } if total > 0) {
            let results = engine
                .search("INC-18372")
                .query("father.md")
                .paths([root.path()])
                .sync(false)
                .lexical_fallback(true)
                .execute()
                .unwrap();
            assert_eq!(
                results.degraded.unwrap().reason,
                seyk_core::DegradedReason::NotYetEmbedded
            );
            assert!(results.hits.iter().any(|hit| hit.signals.filename));
            assert!(results.hits.iter().any(|hit| hit.signals.lexical.is_some()));
            checked.fetch_add(1, Ordering::SeqCst);
        }
    };
    engine
        .index()
        .paths([root.path()])
        .progress(&progress)
        .execute()
        .unwrap();
    assert_eq!(checked.load(Ordering::SeqCst), 1);
}

#[test]
fn skipped_current_versions_do_not_disable_ready_semantic_results() {
    for invalid in [vec![0xff, 0xfe], b"quartz\0invalid".to_vec()] {
        let root = corpus();
        let bad = root.path().join("invalid.txt");
        fs::write(&bad, invalid).unwrap();
        let bad = bad.canonicalize().unwrap();
        let (engine, _) = engine(MemoryCache::new());
        let report = engine.index().paths([root.path()]).execute().unwrap();
        assert_eq!(report.sync.skipped.len(), 1);
        let ready = || {
            engine
                .search("father")
                .paths([root.path()])
                .sync(false)
                .lexical_fallback(true)
                .execute()
                .unwrap()
        };
        let results = ready();
        assert!(results.degraded.is_none());
        assert!(results.hits.iter().any(|h| h.signals.semantic.is_some()));
        assert!(
            results
                .hits
                .iter()
                .all(|h| h.resource.to_file_path().unwrap() != bad)
        );
        // A changed version is pending again, and a later successful attempt
        // clears the skip without keeping the collection in lexical mode.
        fs::write(&bad, "father quartz repaired document").unwrap();
        assert_eq!(
            ready().degraded.unwrap().reason,
            seyk_core::DegradedReason::NotYetEmbedded
        );
        engine.index().paths([root.path()]).execute().unwrap();
        assert!(ready().degraded.is_none());
        assert!(
            ready()
                .hits
                .iter()
                .any(|h| h.resource.to_file_path().unwrap() == bad)
        );
    }
}

#[test]
fn failed_refresh_excludes_old_evidence_with_or_without_previews() {
    for lexical_only in [false, true] {
        for previews in [None, Some(160)] {
            for invalid in [vec![0xff, 0xfe], b"invalid\0bytes".to_vec()] {
                let root = TempDir::new().unwrap();
                let path = root.path().join("quartz.txt");
                fs::write(&path, "quartz lighthouse keeper").unwrap();
                let engine = if lexical_only {
                    SearchEngine::lexical().unwrap()
                } else {
                    engine(MemoryCache::new()).0
                };
                let search = |sync, fallback| {
                    engine
                        .search("quartz")
                        .paths([root.path()])
                        .sync(sync)
                        .lexical_fallback(fallback)
                        .previews(previews)
                        .execute()
                        .unwrap()
                };
                assert!(!search(true, false).hits.is_empty());
                fs::write(&path, invalid).unwrap();
                // Background-indexed callers must not serve the old version
                // even before the new version has been attempted.
                assert!(search(false, true).hits.is_empty());
                assert!(search(true, false).hits.is_empty());
                assert!(search(false, true).hits.is_empty());
                assert!(search(false, false).hits.is_empty());
                fs::write(&path, "quartz restored evidence").unwrap();
                assert!(!search(true, false).hits.is_empty());
            }
        }
    }
}

#[test]
fn reopened_sqlite_index_reestablishes_skips_without_serving_old_bindings() {
    let root = corpus();
    let db = TempDir::new().unwrap();
    let cache_file = db.path().join("cache.sqlite3");
    let path = root.path().join("father.md").canonicalize().unwrap();
    let open = || engine(SqliteCache::open(&cache_file).unwrap()).0;
    let first = open();
    first.index().paths([root.path()]).execute().unwrap();
    drop(first);
    fs::write(&path, [0xff, 0xfe]).unwrap();
    for _ in 0..2 {
        let reopened = open();
        // Daemon/MCP lifetimes run an initial index before reporting ready.
        let report = reopened.index().paths([root.path()]).execute().unwrap();
        assert_eq!(report.sync.skipped.len(), 1);
        let result = reopened
            .search("father")
            .paths([root.path()])
            .sync(false)
            .lexical_fallback(true)
            .previews(None)
            .execute()
            .unwrap();
        assert!(result.degraded.is_none());
        assert!(result.hits.iter().any(|h| h.signals.semantic.is_some()));
        assert!(
            result
                .hits
                .iter()
                .all(|h| h.resource.to_file_path().unwrap() != path)
        );
    }
}

// Canonical provider IDs keep physical temporary paths out of BM25 inputs.
// Compare exact f32 bits, order, locators and existing signal fields.
#[test]
fn ranking_matches_rrf_k5_golden() {
    let mut cases = Vec::new();
    let mut saw_filename = false;
    for sqlite in [false, true] {
        use golden_provider::GoldenProvider;
        use seyk_core::discovery::ResourceKind;
        use seyk_core::segment::TextFormat;
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
        let db = TempDir::new().unwrap();
        let builder = SearchEngine::builder()
            .provider(provider)
            .embedder(FakeEmbedder::new(Dimension::D256, "r1"));
        let engine = if sqlite {
            builder.cache(SqliteCache::open(db.path().join("cache.sqlite3")).unwrap())
        } else {
            builder.cache(MemoryCache::new())
        }
        .build()
        .unwrap();
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Lexical,
        ] {
            for queries in [
                vec!["father"],
                vec!["FATHER.MD"],
                vec!["where did the restore path fail"],
                vec!["INC-18372"],
                vec!["garden.md", "father"],
                vec!["where is the"],
                vec!["absent"],
            ] {
                for limit in [1, 10] {
                    for weight in [None, Some(0.0)] {
                        let mut request = engine
                            .search(queries[0])
                            .mode(mode)
                            .roots([golden_provider::root()])
                            .limit(limit);
                        for q in &queries[1..] {
                            request = request.query(*q);
                        }
                        if let Some(w) = weight {
                            request = request.lexical_weight(w);
                        }
                        let r = request.execute().unwrap();
                        for hit in &r.hits {
                            let name = golden_provider::name(&hit.resource);
                            let named = mode != SearchMode::Semantic
                                && queries.iter().any(|q| {
                                    q.eq_ignore_ascii_case(name)
                                        || q.eq_ignore_ascii_case(name.trim_end_matches(".md"))
                                });
                            assert_eq!(hit.signals.filename, named);
                            assert_eq!(hit.r#match.filename_rank.is_some(), named);
                            saw_filename |= named;
                        }
                        cases.push(serde_json::json!({
                            "mode": mode, "queries": queries, "limit": limit, "weight": weight,
                            "hits": r.hits.iter().map(|h| serde_json::json!({
                                "file": golden_provider::name(&h.resource),
                                "locator": h.locator, "score_bits": h.score.to_bits(),
                                "semantic_bits": h.signals.semantic.map(f32::to_bits),
                                "lexical_bits": h.signals.lexical.map(f32::to_bits),
                                "filename": h.signals.filename, "query_index": h.query_index,
                            })).collect::<Vec<_>>()
                        }));
                    }
                }
            }
        }
    }
    assert!(saw_filename);
    let expected: Vec<serde_json::Value> = include_str!("fixtures/match-ranking-rrf-k5.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(cases, expected);
}

#[test]
fn match_reasons_report_contributing_ranks_and_query_terms() {
    both_caches(|engine, _, root| {
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Lexical,
        ] {
            let r = engine
                .search("father.md")
                .mode(mode)
                .paths([root])
                .limit(1)
                .execute()
                .unwrap();
            let h = &r.hits[0];
            assert_eq!(
                h.r#match.semantic_rank.is_some(),
                h.signals.semantic.is_some()
            );
            assert_eq!(
                h.r#match.keyword_rank.is_some(),
                h.signals.lexical.is_some()
            );
            assert_eq!(h.r#match.filename_rank.is_some(), h.signals.filename);
            if mode == SearchMode::Semantic {
                assert_eq!(h.r#match.semantic_rank, Some(1));
                assert!(h.r#match.matched_terms.is_empty());
            } else {
                assert_eq!(h.r#match.keyword_rank, Some(1));
                assert_eq!(h.r#match.filename_rank, Some(1));
                assert_eq!(h.r#match.matched_terms, ["father", "md"]);
            }
        }
        let disabled = engine
            .search("father.md")
            .paths([root])
            .lexical_weight(0.0)
            .execute()
            .unwrap();
        for h in &disabled.hits {
            assert_eq!(h.r#match.keyword_rank, None);
            assert!(h.r#match.matched_terms.is_empty());
        }
        let r = engine
            .search("INC-18372 absent")
            .query("RESTORE inc-18372")
            .mode(SearchMode::Lexical)
            .paths([root])
            .execute()
            .unwrap();
        assert_eq!(r.hits[0].r#match.matched_terms, ["inc", "18372", "restore"]);
        // An incomplete identifier group must not report its partial match.
        let r = search(&engine, root, "INC-missing restore", SearchMode::Lexical);
        assert_eq!(r.hits[0].r#match.matched_terms, ["restore"]);
        // Old envelopes still deserialize with empty provenance.
        let mut old = serde_json::to_value(&r.hits[0]).unwrap();
        old.as_object_mut().unwrap().remove("match");
        let restored: seyk_core::SearchHit = serde_json::from_value(old).unwrap();
        assert_eq!(restored.r#match, seyk_core::MatchReason::default());
    });
}

#[test]
fn match_terms_use_cached_index_without_reading_source_text() {
    for sqlite in [false, true] {
        let root = TempDir::new().unwrap();
        let file = root.path().join("notes.md");
        fs::write(&file, "Café privatecanary RESTORE").unwrap();
        let db = TempDir::new().unwrap();
        let e = if sqlite {
            engine(SqliteCache::open(db.path().join("index.sqlite")).unwrap()).0
        } else {
            engine(MemoryCache::new()).0
        };
        e.index().paths([root.path()]).execute().unwrap();
        let e = if sqlite {
            drop(e);
            engine(SqliteCache::open(db.path().join("index.sqlite")).unwrap()).0
        } else {
            e
        };
        fs::write(&file, "unrelated replacement").unwrap();
        let r = e
            .search("the CAFÉ restore CAFÉ absent")
            .mode(SearchMode::Lexical)
            .paths([root.path()])
            .sync(false)
            .previews(None)
            .execute()
            .unwrap();
        assert_eq!(r.hits[0].r#match.matched_terms, ["cafe", "restore"]);
        assert!(r.hits[0].preview.is_none());
    }
}

#[test]
fn filename_ranks_preserve_discovery_positions_after_fusion() {
    for sqlite in [false, true] {
        let dir = TempDir::new().unwrap();
        for name in ["a", "b", "c"] {
            fs::create_dir(dir.path().join(name)).unwrap();
            fs::write(dir.path().join(name).join("note.md"), "shared content").unwrap();
        }
        let db = TempDir::new().unwrap();
        let (e, _) = if sqlite {
            engine(SqliteCache::open(db.path().join("cache.db")).unwrap())
        } else {
            engine(MemoryCache::new())
        };
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Lexical,
            SearchMode::Semantic,
        ] {
            let result = e
                .search("note.md")
                .paths([dir.path()])
                .mode(mode)
                .limit(10)
                .execute()
                .unwrap();
            assert_eq!(result.hits.len(), 3);
            for hit in result.hits {
                let path = hit.resource.to_file_path().unwrap();
                let parent = path
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let expected = match parent {
                    "a" => 1,
                    "b" => 2,
                    "c" => 3,
                    _ => unreachable!(),
                };
                assert_eq!(
                    hit.r#match.filename_rank,
                    (mode != SearchMode::Semantic).then_some(expected)
                );
            }
        }
    }
}

#[test]
fn model_free_search_reports_kind_counts_and_completed_progress() {
    let root = corpus();
    fs::write(root.path().join("invalid.txt"), [0xff, 0xfe]).unwrap();
    let engine = SearchEngine::lexical().unwrap();
    let results = engine
        .search("father")
        .paths([root.path()])
        .kind_counts(true)
        .execute()
        .unwrap();
    assert_eq!(
        results.degraded.unwrap().reason,
        seyk_core::DegradedReason::NoModel
    );
    assert_eq!(results.kind_counts.unwrap()["text"], results.hits.len());
    let progress = results.stats.sync.progress.unwrap();
    assert_eq!(progress.by_kind["text"].indexed, 4);
    assert_eq!(progress.by_kind["text"].skipped, 1);
    assert_eq!(progress.by_kind["text"].waiting, 0);
    assert_eq!(progress.eta_seconds, None);
}
