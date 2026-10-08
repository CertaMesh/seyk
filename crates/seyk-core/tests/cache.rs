mod common;

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use common::FakeEmbedder;
use seyk_core::cache::{CacheProfile, CachedResource, CachedSegment, ResourceWrite, SegmentWrite};
use seyk_core::segment::{TextSegmenter, TextSegmenterConfig};
use seyk_core::{
    Dimension, EmbeddingCache, MemoryCache, Modality, ModelIdentity, ResourceId, ResourceLocator,
    ResourceMetadata, ResourceVersion, SearchEngine, SqliteCache,
};
use tempfile::TempDir;

fn shared_engine(
    cache: impl seyk_core::EmbeddingCache + 'static,
    dimension: Dimension,
    revision: &str,
) -> (SearchEngine, std::sync::Arc<FakeEmbedder>) {
    let embedder = std::sync::Arc::new(FakeEmbedder::new(dimension, revision));
    let engine = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .cache(cache)
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 20,
            max_chars: 100,
            overlap_chars: 0,
        }))
        .build()
        .unwrap();
    (engine, embedder)
}

fn search(engine: &SearchEngine, root: &Path) -> seyk_core::SearchResults {
    engine
        .search("alpha garden")
        .paths([root])
        .execute()
        .unwrap()
}

#[test]
fn malformed_vector_bytes_fail_direct_reads_and_search() {
    for remainder in 1..=3 {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("notes.txt");
        fs::write(&file, "alpha garden").unwrap();
        let database = temp.path().join("cache.sqlite3");
        let (engine, fake) =
            shared_engine(SqliteCache::open(&database).unwrap(), Dimension::D128, "r1");
        let hits = search(&engine, &file).hits;
        assert!(!hits.is_empty());
        let wanted = paths(&[&hits[0].resource]);
        let profile = engine.profile_id();
        let cache = SqliteCache::open(&database).unwrap();
        let valid = cache.vectors_for(profile, &wanted).unwrap();
        assert_eq!(valid.len(), 1);
        assert_eq!(valid[0].vector.len(), 128);

        let conn = rusqlite::Connection::open(&database).unwrap();
        let mut bytes: Vec<u8> = conn
            .query_row(
                "SELECT vector FROM embeddings WHERE profile = ?1 AND input_hash = ?2",
                rusqlite::params![profile, valid[0].segment.input_hash],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(bytes.len(), 128 * 4);
        // A full-width vector plus a partial float used to pass the engine's width check.
        bytes.extend(vec![0xff; remainder]);
        assert_eq!(
            conn.execute(
                "UPDATE embeddings SET vector = ?1 WHERE profile = ?2 AND input_hash = ?3",
                rusqlite::params![bytes, profile, valid[0].segment.input_hash],
            )
            .unwrap(),
            1
        );

        let direct_error = cache.vectors_for(profile, &wanted).unwrap_err();
        assert_malformed_vector_error(direct_error);
        fake.reset_resource_count();
        for mode in [
            seyk_core::SearchMode::Semantic,
            seyk_core::SearchMode::Hybrid,
        ] {
            let search_error = engine
                .search("alpha garden")
                .paths([&file])
                .mode(mode)
                .execute()
                .unwrap_err();
            assert_malformed_vector_error(search_error);
        }
        assert_eq!(fake.resource_count(), 0, "read the corrupted cached vector");
    }
}

fn assert_malformed_vector_error(error: seyk_core::Error) {
    match error {
        seyk_core::Error::Cache(rusqlite::Error::FromSqlConversionFailure(
            4,
            rusqlite::types::Type::Blob,
            source,
        )) => assert_eq!(
            source.to_string(),
            "vector byte length must be a multiple of 4"
        ),
        other => panic!("expected a malformed vector cache error, got {other:?}"),
    }
}

#[test]
#[cfg_attr(
    windows,
    ignore = "Solo #4349: timestamp changes need a writable Windows file handle"
)]
fn unchanged_and_mtime_only_changes_reuse_embeddings() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("notes.txt");
    fs::write(&file, "alpha garden\n\nbeta workshop").unwrap();
    let (engine, fake) = shared_engine(MemoryCache::new(), Dimension::D128, "r1");

    search(&engine, temp.path());
    assert!(fake.resource_count() > 0);
    fake.reset_resource_count();
    search(&engine, temp.path());
    assert_eq!(fake.resource_count(), 0);

    fs::File::open(&file)
        .unwrap()
        .set_modified(SystemTime::now() + Duration::from_secs(2))
        .unwrap();
    search(&engine, temp.path());
    assert_eq!(fake.resource_count(), 0);
}

#[test]
fn changed_files_reuse_unchanged_segment_inputs() {
    let temp = TempDir::new().unwrap();
    let first = temp.path().join("first.txt");
    let second = temp.path().join("second.txt");
    fs::write(&first, "alpha garden\n\nbeta workshop").unwrap();
    fs::write(&second, "quiet library\n\norange market").unwrap();
    let (engine, fake) = shared_engine(MemoryCache::new(), Dimension::D128, "r1");
    search(&engine, temp.path());
    fake.reset_resource_count();

    fs::write(&first, "alpha garden\n\nchanged workshop").unwrap();
    search(&engine, temp.path());
    assert_eq!(
        fake.resource_count(),
        1,
        "only the changed segment should be embedded"
    );
}

#[test]
fn model_and_segmenter_profiles_invalidate_vectors() {
    let temp = TempDir::new().unwrap();
    fs::write(
        temp.path().join("notes.txt"),
        "alpha garden\n\nbeta workshop",
    )
    .unwrap();
    let cache = temp.path().join("cache.sqlite3");
    let (first, fake_first) =
        shared_engine(SqliteCache::open(&cache).unwrap(), Dimension::D128, "r1");
    let first_results = search(&first, temp.path());
    assert!(fake_first.resource_count() > 0);
    let first_profile = first.profile_id().to_string();
    drop(first);

    let (changed_dimension, fake_dimension) =
        shared_engine(SqliteCache::open(&cache).unwrap(), Dimension::D256, "r1");
    let second_results = search(&changed_dimension, temp.path());
    assert_ne!(first_profile, changed_dimension.profile_id());
    assert!(fake_dimension.resource_count() > 0);
    assert!(
        second_results.hits.iter().all(|hit| hit
            .resource
            .to_file_path()
            .expect("file resource")
            .exists())
    );
    assert!(
        first_results.hits.iter().all(|hit| hit
            .resource
            .to_file_path()
            .expect("file resource")
            .exists())
    );
    drop(changed_dimension);

    let (changed_revision, fake_revision) =
        shared_engine(SqliteCache::open(&cache).unwrap(), Dimension::D256, "r2");
    search(&changed_revision, temp.path());
    assert!(fake_revision.resource_count() > 0);
    drop(changed_revision);

    let embedder = std::sync::Arc::new(FakeEmbedder::new(Dimension::D256, "r2"));
    let changed_segmenter = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .cache(SqliteCache::open(&cache).unwrap())
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 10,
            max_chars: 100,
            overlap_chars: 0,
        }))
        .build()
        .unwrap();
    search(&changed_segmenter, temp.path());
    assert!(embedder.resource_count() > 0);
}

#[test]
fn index_removes_deleted_resources_and_locators_map_to_source() {
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("notes.txt");
    fs::write(
        &file,
        "first line\nalpha garden\nthird line\n\nbeta workshop",
    )
    .unwrap();
    let (engine, _) = shared_engine(MemoryCache::new(), Dimension::D128, "r1");
    let results = search(&engine, temp.path());
    let hit = results.hits.first().unwrap();
    let ResourceLocator::Text {
        byte_start,
        byte_end,
        line_start,
        line_end,
    } = &hit.locator
    else {
        panic!("expected text locator")
    };
    let source = fs::read_to_string(&file).unwrap();
    assert!(source[*byte_start as usize..*byte_end as usize].contains("alpha garden"));
    assert_eq!((*line_start, *line_end), (Some(1), Some(3)));

    fs::remove_file(&file).unwrap();
    let report = engine.index().paths([temp.path()]).execute().unwrap();
    assert_eq!(report.resources_removed, 1);
    assert!(search(&engine, temp.path()).hits.is_empty());
}

#[test]
fn memory_cache_is_ephemeral_and_sqlite_is_private() {
    let temp = TempDir::new().unwrap();
    fs::write(temp.path().join("notes.txt"), "alpha garden").unwrap();
    let (memory, _) = shared_engine(MemoryCache::new(), Dimension::D128, "r1");
    search(&memory, temp.path());
    assert_eq!(
        memory
            .with_cache(|cache| Ok(cache.stats()?.location))
            .unwrap(),
        None
    );

    let cache_path = temp.path().join("private.sqlite3");
    let cache = SqliteCache::open(&cache_path).unwrap();
    assert_eq!(cache.path(), cache_path);
    #[cfg(unix)]
    assert_eq!(
        std::os::unix::fs::PermissionsExt::mode(&fs::metadata(cache.path()).unwrap().permissions())
            & 0o777,
        0o600
    );
}

fn profile(revision: &str) -> CacheProfile {
    CacheProfile {
        model: ModelIdentity {
            model_id: "test/model".into(),
            revision: revision.into(),
            variant: "fake".into(),
            dimension: Dimension::D128,
            normalization: "l2".into(),
            query_prompt: String::new(),
            document_prompt: String::new(),
            max_tokens: 64,
        },
        segmenter_version: "1".into(),
    }
}

fn resource(hashes: &[&str]) -> CachedResource {
    CachedResource {
        descriptor_hash: Some("descriptor".into()),
        version: ResourceVersion::known("fs:1:1"),
        metadata: ResourceMetadata {
            media_type: None,
            modality: Modality::Text,
            size_bytes: 1,
            modified_ns: Some(1),
            content_hash: hashes.join("+"),
        },
        segments: hashes
            .iter()
            .map(|h| CachedSegment {
                locator: ResourceLocator::WholeResource,
                heading: None,
                input_hash: (*h).to_string(),
            })
            .collect(),
    }
}

fn paths(items: &[&ResourceId]) -> HashSet<ResourceId> {
    items.iter().map(|p| (*p).clone()).collect()
}

fn hashes(items: &[&str]) -> Vec<String> {
    items.iter().map(|h| (*h).to_string()).collect()
}

#[test]
fn memory_cache_keeps_profiles_apart_for_the_same_path_and_input() {
    let mut cache = MemoryCache::new();
    let path = &ResourceId::from_path(std::env::temp_dir().join("notes.txt")).unwrap();
    for (id, value) in [("a", 1.0), ("b", 2.0)] {
        cache.register_profile(id, &profile(id)).unwrap();
        cache
            .put_embeddings(id, &[("h".into(), vec![value])])
            .unwrap();
        cache
            .put_resource(id, path, &write(&resource(&["h"]), ""))
            .unwrap();
    }

    for (id, value) in [("a", 1.0), ("b", 2.0)] {
        let found = cache.vectors_for(id, &paths(&[path])).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].vector, vec![value]);
    }

    let mut metadata = resource(&["h"]).metadata;
    metadata.modified_ns = Some(7);
    cache
        .touch_resource("a", path, &metadata, &ResourceVersion::known("fs:1:7"))
        .unwrap();
    cache
        .remove_resources("b", std::slice::from_ref(path))
        .unwrap();
    assert_eq!(cache.prune_orphans("b").unwrap(), 1);
    assert_eq!(
        cache
            .resource("a", path)
            .unwrap()
            .unwrap()
            .metadata
            .modified_ns,
        Some(7)
    );
    assert_eq!(cache.known_inputs("a", &hashes(&["h"])).unwrap().len(), 1);
    assert!(cache.known_inputs("b", &hashes(&["h"])).unwrap().is_empty());

    let mut stats = cache.stats().unwrap().profiles;
    stats.sort_by(|x, y| x.profile_id.cmp(&y.profile_id));
    let counts: Vec<_> = stats
        .iter()
        .map(|s| (s.profile_id.as_str(), s.resources, s.segments, s.embeddings))
        .collect();
    assert_eq!(counts, [("a", 1, 1, 1), ("b", 0, 0, 0)]);
}

#[test]
fn memory_cache_accepts_writes_before_registration() {
    let mut cache = MemoryCache::new();
    let path = &ResourceId::from_path(std::env::temp_dir().join("notes.txt")).unwrap();
    // Batching writes vectors before their resource, so a vector without a
    // segment is known but not yet searchable.
    cache
        .put_embeddings("p", &[("h".into(), vec![1.0])])
        .unwrap();
    assert_eq!(
        cache.known_inputs("p", &hashes(&["h", "x"])).unwrap().len(),
        1
    );
    assert!(cache.vectors_for("p", &paths(&[path])).unwrap().is_empty());
    assert!(cache.stats().unwrap().profiles.is_empty());

    cache
        .put_resource("p", path, &write(&resource(&["h"]), ""))
        .unwrap();
    cache.register_profile("p", &profile("r1")).unwrap();
    assert_eq!(cache.vectors_for("p", &paths(&[path])).unwrap().len(), 1);
    let stats = cache.stats().unwrap().profiles;
    assert_eq!(
        (stats[0].resources, stats[0].segments, stats[0].embeddings),
        (1, 1, 1)
    );

    cache.clear().unwrap();
    assert!(cache.resource_index("p").unwrap().is_empty());
    assert!(cache.known_inputs("p", &hashes(&["h"])).unwrap().is_empty());
}

#[test]
fn memory_cache_prunes_only_unreferenced_vectors() {
    let mut cache = MemoryCache::new();
    let (first, second) = (
        &ResourceId::from_path(std::env::temp_dir().join("first.txt")).unwrap(),
        &ResourceId::from_path(std::env::temp_dir().join("second.txt")).unwrap(),
    );
    cache
        .put_embeddings(
            "p",
            &[
                ("shared".into(), vec![1.0]),
                ("own".into(), vec![2.0]),
                ("orphan".into(), vec![3.0]),
            ],
        )
        .unwrap();
    cache
        .put_embeddings("other", &[("orphan".into(), vec![3.0])])
        .unwrap();
    cache
        .put_resource("p", first, &write(&resource(&["shared", "own"]), ""))
        .unwrap();
    cache
        .put_resource("p", second, &write(&resource(&["shared"]), ""))
        .unwrap();

    let only_second = cache.vectors_for("p", &paths(&[second])).unwrap();
    assert_eq!(only_second.len(), 1);
    assert_eq!(&only_second[0].resource, second);

    assert_eq!(cache.prune_orphans("p").unwrap(), 1);
    // A vector still used by another resource survives its sibling's removal.
    cache
        .remove_resources("p", std::slice::from_ref(first))
        .unwrap();
    assert_eq!(cache.prune_orphans("p").unwrap(), 1);
    assert_eq!(
        cache
            .known_inputs("p", &hashes(&["shared", "own"]))
            .unwrap(),
        HashSet::from(["shared".to_string()])
    );
    assert_eq!(
        cache
            .known_inputs("other", &hashes(&["orphan"]))
            .unwrap()
            .len(),
        1
    );
    assert_eq!(cache.prune_orphans("absent").unwrap(), 0);
}

#[test]
fn characterize_cross_profile_bm25_score_and_top_k_contamination() {
    use seyk_core::cache::LexicalHit;
    use seyk_core::lexical::LexicalQuery;

    // F-S04-2 / Solo #4390 (P10): characterize the existing defect, not a desired
    // scoring contract. P11 should replace the contaminated expectations with
    // profile isolation. No production ranking or schema change belongs here.
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("cache.sqlite3");
    let mut sqlite = SqliteCache::open(&database).unwrap();
    let mut memory = MemoryCache::new();
    let ids: Vec<ResourceId> = (0..18)
        .map(|i| format!("mem://corpus/d{i:02}").parse().unwrap())
        .collect();
    let wanted: HashSet<_> = ids[..12].iter().cloned().collect();
    let query = LexicalQuery::new(["quartz amber"]);
    let mut bodies = vec![
        "quartz ".repeat(6) + "amber",
        "quartz amber amber amber".into(),
    ];
    bodies.extend((1..=10).map(|n| "neutral ".repeat(n)));
    // Both query terms occur in 2/12 documents initially. Six long quartz-only
    // B documents make that 8/18 vs 2/18: neither IDF hits the floor. URI labels
    // have equal token lengths and contain neither query term on any platform.
    let value = resource(&["synthetic"]);
    for cache in [&mut sqlite as &mut dyn EmbeddingCache, &mut memory] {
        for p in ["a", "b"] {
            cache.register_profile(p, &profile(p)).unwrap();
        }
        for (id, body) in ids.iter().zip(&bodies) {
            cache.put_resource("a", id, &write(&value, body)).unwrap();
        }
    }
    let measure = |cache: &dyn EmbeddingCache| {
        let hits = cache.lexical_search("a", &query, &wanted, 12).unwrap();
        assert_eq!(hits.len(), 2);
        for k in [1, 2] {
            assert_eq!(
                cache.lexical_search("a", &query, &wanted, k).unwrap(),
                hits[..k]
            );
        }
        assert!(
            hits.iter()
                .all(|hit| wanted.contains(&hit.resource) && hit.ordinal == 0)
        );
        hits
    };
    let same_scores_and_order = |left: &[LexicalHit], right: &[LexicalHit]| {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right) {
            assert_eq!((&l.resource, l.ordinal), (&r.resource, r.ordinal));
            assert!((l.score - r.score).abs() < 1e-6, "{l:?} != {r:?}");
        }
    };
    let alone = measure(&sqlite);
    let memory_alone = measure(&memory);
    same_scores_and_order(&alone, &memory_alone);
    assert_eq!(alone[0].resource, ids[0]);
    assert_eq!(alone[1].resource, ids[1]);
    assert!(alone[0].score > alone[1].score);

    for cache in [&mut sqlite as &mut dyn EmbeddingCache, &mut memory] {
        for (i, id) in ids[12..].iter().enumerate() {
            cache
                .put_resource("b", id, &write(&value, &"quartz ".repeat(40 + i)))
                .unwrap();
        }
    }
    let contaminated = measure(&sqlite);
    same_scores_and_order(&memory_alone, &measure(&memory));
    assert_eq!(contaminated[0].resource, ids[1]);
    assert_eq!(contaminated[1].resource, ids[0]);
    assert!(contaminated[0].score > contaminated[1].score);
    for original in &alone {
        let changed = contaminated
            .iter()
            .find(|h| h.resource == original.resource)
            .unwrap();
        let delta = changed.score - original.score;
        assert!(
            delta.abs() > 0.1,
            "expected measurable movement for {original:?}"
        );
        println!(
            "P10 {}: A alone / MemoryCache={:.6}, with B={:.6}, delta={delta:+.6} ({:+.2}%)",
            original.resource,
            original.score,
            changed.score,
            100.0 * delta / original.score,
        );
    }

    // Read the existing contentless index through its supported vocabulary API.
    // Per-instance postings retain the profile join, frequencies and lengths
    // needed to investigate P11 without re-reading source bodies. This only
    // measures the index; it neither migrates nor changes the scoring path.
    let conn = rusqlite::Connection::open(&database).unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE temp.p10_vocab USING fts5vocab(main, segments_fts, instance);",
    )
    .unwrap();
    let counts = || -> Vec<(String, i64, i64, i64)> {
        conn.prepare(
            "SELECT s.profile, count(DISTINCT s.rowid), count(*),
                count(DISTINCT CASE WHEN v.term = 'quartz' THEN s.rowid END)
         FROM p10_vocab v JOIN segments s ON s.rowid = v.doc
         GROUP BY s.profile ORDER BY s.profile",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
    };
    assert_eq!(
        counts(),
        [("a".into(), 12, 114, 2), ("b".into(), 6, 279, 6)]
    );

    for cache in [&mut sqlite as &mut dyn EmbeddingCache, &mut memory] {
        cache.remove_resources("b", &ids[12..]).unwrap();
        assert!(cache.resource_index("b").unwrap().is_empty());
        // A's complete stored resources stayed fixed throughout the experiment.
        for id in &ids[..12] {
            assert_eq!(cache.resource("a", id).unwrap().unwrap(), value);
        }
    }
    let removed = measure(&sqlite);
    assert_eq!(counts(), [("a".into(), 12, 114, 2)]);
    // Bundled SQLite 3.53.2's contentless-delete path adds tombstones without
    // decrementing the stored row/token totals used by BM25. Live term matches
    // exclude B, but those totals still include it: deleting B restores the order,
    // NOT the scores, even after reopening. P11 must account for this lifecycle
    // effect instead of assuming profile deletion restores shared statistics.
    for (original, deleted) in alone.iter().zip(&removed) {
        assert_eq!(original.resource, deleted.resource);
        let delta = deleted.score - original.score;
        assert!(delta > 1.0);
        println!(
            "P10 {}: after deleting B={:.6}, delta from A alone={delta:+.6} ({:+.2}%)",
            deleted.resource,
            deleted.score,
            100.0 * delta / original.score,
        );
    }
    same_scores_and_order(&removed, &measure(&SqliteCache::open(&database).unwrap()));
    same_scores_and_order(&memory_alone, &measure(&memory));
    println!(
        "P10 top-1: d00 -> d01 -> d00; top-2 order reverses with B; deletion does not restore scores"
    );
}

#[test]
fn lexical_ties_use_path_bytes_then_ordinal_before_limit_in_both_caches() {
    let names = [
        "/notes/zulu.md",
        "/notes/alpha.md",
        "/notes/delta.md",
        "/notes/bravo.md",
    ]
    .map(|name| std::env::temp_dir().join(name.trim_start_matches('/')));
    let wanted: HashSet<ResourceId> = names
        .iter()
        .map(|p| ResourceId::from_path(p).unwrap())
        .collect();
    let query = seyk_core::lexical::LexicalQuery::new(["quartz"]);
    let check = |cache: &mut dyn EmbeddingCache| {
        cache.register_profile("ties", &profile("ties")).unwrap();
        for name in &names {
            let r = resource(&["first", "second"]);
            cache
                .put_resource(
                    "ties",
                    &ResourceId::from_path(name).unwrap(),
                    &write(&r, "quartz pebble"),
                )
                .unwrap();
        }
        let hits = cache.lexical_search("ties", &query, &wanted, 100).unwrap();
        assert_eq!(hits.len(), 8);
        assert!(hits.iter().all(|h| h.score == hits[0].score));
        let order: Vec<_> = hits.into_iter().map(|h| (h.resource, h.ordinal)).collect();
        for limit in [0, 1, 3, 8] {
            let limited: Vec<_> = cache
                .lexical_search("ties", &query, &wanted, limit)
                .unwrap()
                .into_iter()
                .map(|h| (h.resource, h.ordinal))
                .collect();
            assert_eq!(limited, order[..limit]);
        }
        order
    };
    let db = TempDir::new().unwrap();
    let expected = check(&mut SqliteCache::open(db.path().join("cache.sqlite3")).unwrap());
    let mut sorted = expected.clone();
    sorted.sort_by(|a, b| {
        a.0.to_file_path()
            .unwrap()
            .as_os_str()
            .as_encoded_bytes()
            .cmp(b.0.to_file_path().unwrap().as_os_str().as_encoded_bytes())
            .then(a.1.cmp(&b.1))
    });
    assert_eq!(expected, sorted);
    for _ in 0..20 {
        assert_eq!(check(&mut MemoryCache::new()), expected);
    }
}

#[test]
fn v3_upgrade_keeps_file_keys_vectors_segments_and_keywords_without_embedding() {
    use rusqlite::Connection;
    let notes = TempDir::new().unwrap();
    let db = TempDir::new().unwrap();
    fs::write(notes.path().join("alpha.md"), "quartz pebble alpha garden").unwrap();
    fs::write(notes.path().join("beta.md"), "quartz pebble beta workshop").unwrap();
    let file = db.path().join("cache.sqlite3");
    let (engine, _) = shared_engine(SqliteCache::open(&file).unwrap(), Dimension::D128, "r1");
    let before = engine
        .search("quartz")
        .paths([notes.path()])
        .mode(seyk_core::SearchMode::Lexical)
        .execute()
        .unwrap();
    drop(engine);
    let conn = Connection::open(&file).unwrap();
    let snapshot = |conn: &Connection| {
        let vectors = conn
            .prepare("SELECT profile,input_hash,vector FROM embeddings ORDER BY profile,input_hash")
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let segments = conn
            .prepare(
                "SELECT rowid,path,ordinal,locator,heading,input_hash FROM segments ORDER BY rowid",
            )
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let fts = conn
            .prepare("SELECT id,block FROM segments_fts_data ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        (vectors, segments, fts)
    };
    // Recreate the actual v3 resources layout, without provider state.
    conn.execute_batch("ALTER TABLE resources DROP COLUMN version; ALTER TABLE resources DROP COLUMN descriptor_hash; UPDATE meta SET value='3' WHERE key='schema_version';").unwrap();
    let stored = snapshot(&conn);
    drop(conn);
    let cache = SqliteCache::open(&file).unwrap();
    let conn = Connection::open(&file).unwrap();
    assert_eq!(snapshot(&conn), stored); // Migration preserves every existing byte/rowid.
    drop(conn);
    let (engine, fake) = shared_engine(cache, Dimension::D128, "r1");
    let after = engine
        .search("quartz")
        .paths([notes.path()])
        .mode(seyk_core::SearchMode::Lexical)
        .execute()
        .unwrap();
    assert_eq!(fake.resource_count(), 0);
    assert_eq!(after.stats.sync.resources_updated, 2);
    assert_eq!(after.hits, before.hits);
    drop(engine);
    let conn = Connection::open(&file).unwrap();
    assert_eq!(snapshot(&conn).0, stored.0);
    assert_eq!(
        conn.query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "5"
    );
    drop(conn);
    let (engine, fake) = shared_engine(SqliteCache::open(&file).unwrap(), Dimension::D128, "r1");
    engine
        .index()
        .paths([notes.path().join("alpha.md")])
        .execute()
        .unwrap();
    assert_eq!(fake.resource_count(), 0);
    assert_eq!(
        engine
            .with_cache(|c| Ok(c.resource_index(engine.profile_id())?.len()))
            .unwrap(),
        2
    );
}

#[test]
fn both_caches_store_non_file_identities_and_opaque_versions() {
    fn check(cache: &mut dyn EmbeddingCache) {
        let id: ResourceId = "mem://book/note.txt".parse().unwrap();
        cache.register_profile("p", &profile("r1")).unwrap();
        let mut value = resource(&["generic"]);
        value.version = ResourceVersion::known("opaque-revision-42");
        cache
            .put_embeddings("p", &[("generic".into(), vec![1.0])])
            .unwrap();
        cache
            .put_resource("p", &id, &write(&value, "quartz pebble"))
            .unwrap();
        let found = cache.resource("p", &id).unwrap().unwrap();
        assert_eq!(found.version, value.version);
        assert_eq!(found.descriptor_hash, value.descriptor_hash);
        assert_eq!(
            cache.resource_index("p").unwrap()[&id].descriptor_hash,
            value.descriptor_hash
        );
        assert_eq!(found.segments, value.segments);
        let wanted = HashSet::from([id.clone()]);
        assert_eq!(cache.vectors_for("p", &wanted).unwrap()[0].resource, id);
        let q = seyk_core::lexical::LexicalQuery::new(["quartz"]);
        assert_eq!(
            cache.lexical_search("p", &q, &wanted, 1).unwrap()[0].resource,
            id
        );
        cache
            .touch_resource("p", &id, &value.metadata, &ResourceVersion::known("next"))
            .unwrap();
        assert_eq!(
            cache.resource_index("p").unwrap()[&id].version,
            ResourceVersion::known("next")
        );
    }
    check(&mut MemoryCache::new());
    let temp = TempDir::new().unwrap();
    let file = temp.path().join("generic.sqlite3");
    check(&mut SqliteCache::open(&file).unwrap());
    let cache = SqliteCache::open(&file).unwrap();
    let id: ResourceId = "mem://book/note.txt".parse().unwrap();
    assert_eq!(
        cache
            .resource("p", &id)
            .unwrap()
            .unwrap()
            .descriptor_hash
            .as_deref(),
        Some("descriptor")
    );
    assert_eq!(
        cache.resource("p", &id).unwrap().unwrap().version,
        ResourceVersion::known("next")
    );
}

fn write<'a>(resource: &'a CachedResource, lexical_body: &'a str) -> ResourceWrite<'a> {
    ResourceWrite {
        metadata: &resource.metadata,
        version: &resource.version,
        descriptor_hash: resource.descriptor_hash.as_deref(),
        segments: resource
            .segments
            .iter()
            .map(|segment| SegmentWrite {
                segment,
                lexical_body,
            })
            .collect(),
    }
}

#[test]
fn explicit_bodies_replace_keywords_without_persisting_source() {
    fn check(cache: &mut dyn EmbeddingCache, permits_unregistered: bool) {
        let id: ResourceId = "mem://notes/plain".parse().unwrap();
        let image_id: ResourceId = "mem://notes/diagram.png".parse().unwrap();
        let wanted = HashSet::from([id.clone(), image_id.clone()]);
        let query = |word| seyk_core::lexical::LexicalQuery::new([word]);
        let value = resource(&["text"]);
        let mut picture = resource(&["image"]);
        picture.metadata.modality = Modality::Image;
        // Preserve the established contract: MemoryCache accepts pre-registration
        // writes; SQLite rejects them atomically through its foreign key.
        let early = cache.put_resource("first", &id, &write(&value, "quartz pebble"));
        assert_eq!(early.is_ok(), permits_unregistered);
        assert_eq!(
            cache.resource("first", &id).unwrap().is_some(),
            permits_unregistered
        );
        cache.register_profile("first", &profile("first")).unwrap();
        cache
            .register_profile("second", &profile("second"))
            .unwrap();
        cache
            .put_resource("first", &id, &write(&value, "quartz pebble"))
            .unwrap();
        cache
            .put_resource("first", &image_id, &write(&picture, ""))
            .unwrap();
        cache
            .put_resource("second", &id, &write(&value, "granite"))
            .unwrap();
        let read = cache.resource("first", &id).unwrap().unwrap();
        assert_eq!(read, value);
        let serialized = serde_json::to_string(&read.segments).unwrap();
        assert!(!serialized.contains("quartz"));
        assert!(!serialized.contains("lexical"));
        assert_eq!(
            cache
                .lexical_search("first", &query("quartz"), &wanted, 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            cache
                .lexical_search("first", &query("diagram"), &wanted, 10)
                .unwrap()[0]
                .resource,
            image_id
        );
        cache
            .put_resource("first", &id, &write(&read, "basalt"))
            .unwrap();
        assert!(
            cache
                .lexical_search("first", &query("quartz"), &wanted, 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            cache
                .lexical_search("first", &query("basalt"), &wanted, 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            cache
                .lexical_search("second", &query("granite"), &wanted, 10)
                .unwrap()
                .len(),
            1
        );
        cache
            .remove_resources("first", std::slice::from_ref(&id))
            .unwrap();
        assert!(
            cache
                .lexical_search("first", &query("basalt"), &wanted, 10)
                .unwrap()
                .is_empty()
        );
        cache.clear().unwrap();
        assert!(cache.resource_index("second").unwrap().is_empty());
        assert!(
            cache
                .lexical_search("first", &query("diagram"), &wanted, 10)
                .unwrap()
                .is_empty()
        );
    }
    check(&mut MemoryCache::new(), true);
    let temp = TempDir::new().unwrap();
    check(
        &mut SqliteCache::open(temp.path().join("writes.sqlite3")).unwrap(),
        false,
    );
}

#[test]
fn profile_serialization_and_hash_match_the_pre_split_binary() {
    let json = include_str!("fixtures/pre_split_profile.json");
    let profile: CacheProfile = serde_json::from_str(json).unwrap();
    assert_eq!(serde_json::to_string(&profile).unwrap(), json);
    assert_eq!(profile.id(), "b14d0b2f2629a98e1614b142");
}

#[test]
fn copied_v4_migration_preserves_vectors_fts_and_warm_reuse() {
    use rusqlite::Connection;
    let dir = TempDir::new().unwrap();
    let notes = dir.path().join("notes");
    fs::create_dir(&notes).unwrap();
    fs::write(notes.join("note.txt"), "alpha garden\n\nbeta workshop").unwrap();
    let v4 = dir.path().join("v4.db");
    let (engine, _) = shared_engine(SqliteCache::open(&v4).unwrap(), Dimension::D128, "r1");
    let before = search(&engine, &notes);
    drop(engine);
    let conn = Connection::open(&v4).unwrap();
    conn.execute_batch("ALTER TABLE resources DROP COLUMN media_type; UPDATE meta SET value='4' WHERE key='schema_version'; PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
    let snapshot = |c: &Connection| {
        ["SELECT quote(profile)||quote(input_hash)||hex(vector) FROM embeddings ORDER BY profile,input_hash",
         "SELECT quote(rowid)||quote(path)||quote(ordinal)||quote(locator)||quote(heading)||quote(input_hash) FROM segments ORDER BY rowid",
         "SELECT quote(id)||hex(block) FROM segments_fts_data ORDER BY id",
         "SELECT quote(id)||quote(profile)||quote(dimension) FROM profiles ORDER BY id"]
        .map(|sql| c.prepare(sql).unwrap().query_map([],|r|r.get::<_,String>(0)).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap())
    };
    let stored = snapshot(&conn);
    drop(conn);
    let copy = dir.path().join("v5-copy.db");
    fs::copy(&v4, &copy).unwrap();
    let (engine, fake) = shared_engine(SqliteCache::open(&copy).unwrap(), Dimension::D128, "r1");
    let conn = Connection::open(&copy).unwrap();
    assert_eq!(snapshot(&conn), stored);
    assert_eq!(search(&engine, &notes).hits, before.hits);
    assert_eq!(fake.resource_count(), 0);
    assert_eq!(snapshot(&conn), stored);
    assert_eq!(
        Connection::open(&v4)
            .unwrap()
            .query_row(
                "SELECT value FROM meta WHERE key='schema_version'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        "4"
    );
}
