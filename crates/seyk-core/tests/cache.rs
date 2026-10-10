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
fn cross_profile_bm25_scores_and_top_k_are_isolated() {
    use seyk_core::cache::LexicalHit;
    use seyk_core::lexical::LexicalQuery;

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
        cache.register_profile("a", &profile("a")).unwrap();
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
        cache.register_profile("b", &profile("b")).unwrap();
        same_scores_and_order(&alone, &measure(cache));
        for (i, id) in ids[12..].iter().enumerate() {
            cache
                .put_resource("b", id, &write(&value, &"quartz ".repeat(40 + i)))
                .unwrap();
        }
    }
    same_scores_and_order(&alone, &measure(&sqlite));
    same_scores_and_order(&memory_alone, &measure(&memory));
    for cache in [&mut sqlite as &mut dyn EmbeddingCache, &mut memory] {
        cache
            .put_resource(
                "b",
                &ids[12],
                &write(&resource(&["one", "two"]), "amber amber"),
            )
            .unwrap();
    }
    same_scores_and_order(&alone, &measure(&sqlite));
    for cache in [&mut sqlite as &mut dyn EmbeddingCache, &mut memory] {
        cache.remove_resources("b", &ids[12..]).unwrap();
        cache.prune_orphans("b").unwrap();
        for id in &ids[..12] {
            assert_eq!(cache.resource("a", id).unwrap().unwrap(), value);
        }
    }
    same_scores_and_order(&alone, &measure(&sqlite));
    same_scores_and_order(&alone, &measure(&SqliteCache::open(&database).unwrap()));
    same_scores_and_order(&memory_alone, &measure(&memory));
    println!(
        "P11 top-1/top-2 unchanged after B add/replace/delete/prune/reopen; scores {:.6}, {:.6}",
        alone[0].score, alone[1].score
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
        let fts = common::legacy::postings(conn);
        (vectors, segments, fts)
    };
    common::legacy::downgrade(&conn, "3");
    let stored = snapshot(&conn);
    drop(conn);
    let cache = SqliteCache::open(&file).unwrap();
    let conn = Connection::open(&file).unwrap();
    assert_eq!(snapshot(&conn), stored); // Preserve vector bytes, segment rowids and every posting.
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
        "6"
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
    common::legacy::downgrade(&conn, "4");
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    let snapshot = |c: &Connection| {
        ["SELECT quote(profile)||quote(input_hash)||hex(vector) FROM embeddings ORDER BY profile,input_hash",
         "SELECT quote(rowid)||quote(path)||quote(ordinal)||quote(locator)||quote(heading)||quote(input_hash) FROM segments ORDER BY rowid",
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

fn fts_totals(conn: &rusqlite::Connection, profile: &str) -> Vec<u8> {
    conn.query_row(
        &format!(
            "SELECT block FROM {}_data WHERE id=1",
            common::legacy::table(profile)
        ),
        [],
        |r| r.get(0),
    )
    .unwrap()
}

fn assert_integrity(conn: &rusqlite::Connection, profile: &str) {
    let table = common::legacy::table(profile);
    conn.execute_batch(&format!(
        "INSERT INTO {table}({table},rank) VALUES('integrity-check',1)"
    ))
    .unwrap();
}

#[test]
fn replacements_and_deletions_match_a_fresh_live_corpus() {
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("churn.db");
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.register_profile("p", &profile("p")).unwrap();
    let ids: Vec<ResourceId> = (0..12)
        .map(|i| format!("mem://corpus/d{i:02}").parse().unwrap())
        .collect();
    let query = seyk_core::lexical::LexicalQuery::new(["quartz amber"]);
    let wanted = ids.iter().cloned().collect();
    let mut live = std::collections::BTreeMap::new();
    for (i, id) in ids.iter().enumerate() {
        let body = if i < 2 {
            "quartz quartz amber"
        } else {
            "neutral pebble"
        };
        cache
            .put_resource("p", id, &write(&resource(&["a", "b"]), body))
            .unwrap();
        live.insert(i, (2, body));
    }
    // Fewer segments, empty body, no segments, duplicate removals, rowid reuse,
    // delete-to-empty and reinsert. Compare immediately AND on a fresh connection.
    for step in 0..24 {
        let i = step % 12;
        if step % 3 == 0 {
            cache
                .remove_resources("p", &[ids[i].clone(), ids[i].clone()])
                .unwrap();
            live.remove(&i);
        } else {
            let count = step % 3 - 1;
            let body = if step % 2 == 0 {
                ""
            } else {
                "quartz amber amber"
            };
            let hashes = vec!["replacement"; count];
            cache
                .put_resource("p", &ids[i], &write(&resource(&hashes), body))
                .unwrap();
            live.insert(i, (count, body));
        }
        let fresh_db = dir.path().join(format!("fresh-{step}.db"));
        let mut fresh = SqliteCache::open(&fresh_db).unwrap();
        let mut memory = MemoryCache::new();
        for c in [&mut fresh as &mut dyn EmbeddingCache, &mut memory] {
            c.register_profile("p", &profile("p")).unwrap();
            for (&i, &(count, body)) in &live {
                c.put_resource(
                    "p",
                    &ids[i],
                    &write(&resource(&vec!["replacement"; count]), body),
                )
                .unwrap();
            }
        }
        let expected = fresh.lexical_search("p", &query, &wanted, 12).unwrap();
        for c in [
            &cache as &dyn EmbeddingCache,
            &SqliteCache::open(&database).unwrap(),
            &memory,
        ] {
            assert_eq!(
                c.lexical_search("p", &query, &wanted, 12).unwrap(),
                expected,
                "step {step}"
            );
        }
        let conn = rusqlite::Connection::open(&database).unwrap();
        assert_eq!(
            fts_totals(&conn, "p"),
            fts_totals(&rusqlite::Connection::open(&fresh_db).unwrap(), "p")
        );
        assert_integrity(&conn, "p");
    }
    cache.remove_resources("p", &ids).unwrap();
    cache.remove_resources("p", &ids).unwrap();
    assert!(
        cache
            .lexical_search("p", &query, &wanted, 12)
            .unwrap()
            .is_empty()
    );
    cache
        .put_resource("p", &ids[0], &write(&resource(&["again"]), "quartz amber"))
        .unwrap();
    assert_eq!(
        cache.lexical_search("p", &query, &wanted, 1).unwrap().len(),
        1
    );
    assert_integrity(&rusqlite::Connection::open(&database).unwrap(), "p");
}

#[test]
fn arbitrary_profiles_lifecycle_missing_tables_and_path_subset() {
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("profiles.db");
    let mut cache = SqliteCache::open(&database).unwrap();
    let id: ResourceId = "mem://corpus/note".parse().unwrap();
    let second: ResourceId = "mem://corpus/second".parse().unwrap();
    let wanted = paths(&[&second]);
    let query = seyk_core::lexical::LexicalQuery::new(["quartz"]);
    for p in ["", "a", "A", "a_data", "'\";--", "Σ東京"] {
        cache.register_profile(p, &profile(p)).unwrap();
        cache.register_profile(p, &profile(p)).unwrap();
        assert!(
            cache
                .lexical_search(p, &query, &wanted, 1)
                .unwrap()
                .is_empty()
        );
        cache
            .put_resource(p, &id, &write(&resource(&["a"]), "quartz quartz"))
            .unwrap();
        cache
            .put_resource(p, &second, &write(&resource(&["b"]), "quartz"))
            .unwrap();
        assert_eq!(
            cache.lexical_search(p, &query, &wanted, 1).unwrap()[0].resource,
            second
        );
    }
    let conn = rusqlite::Connection::open(&database).unwrap();
    assert!(
        cache
            .lexical_search("unknown", &query, &wanted, 1)
            .unwrap()
            .is_empty()
    );
    cache
        .remove_resources("unknown", std::slice::from_ref(&id))
        .unwrap();
    conn.execute_batch("DROP TABLE segments_fts_p_61").unwrap();
    assert!(cache.lexical_search("a", &query, &wanted, 1).is_err());
    drop(cache);
    let mut cache = SqliteCache::open(&database).unwrap();
    let error = cache.register_profile("a", &profile("a")).unwrap_err();
    assert!(error.to_string().contains("corrupt cache"));
    assert_eq!(cache.resource_index("a").unwrap().len(), 2);
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name = 'segments_fts_p_61'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    conn.execute_batch("CREATE VIRTUAL TABLE segments_fts_p_deadbeef USING fts5(body)")
        .unwrap();
    cache.clear().unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name GLOB 'segments_fts_p_*'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    cache.register_profile("a", &profile("a")).unwrap();
    cache
        .put_resource("a", &id, &write(&resource(&["a"]), "quartz"))
        .unwrap();
    assert_eq!(cache.stats().unwrap().profiles[0].segments, 1);
}

#[test]
fn v5_migration_preserves_postings_sparse_ids_vectors_and_empty_documents() {
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("v5.db");
    let mut cache = SqliteCache::open(&database).unwrap();
    let ids: Vec<ResourceId> = (0..12)
        .map(|i| format!("mem://zeta/zeta{i}").parse().unwrap())
        .collect();
    for p in ["a", "東京"] {
        cache.register_profile(p, &profile(p)).unwrap();
        cache
            .put_embeddings(
                p,
                &[
                    ("h".into(), vec![1.25, -0.5]),
                    ("orphan".into(), vec![0.75]),
                ],
            )
            .unwrap();
        for (i, id) in ids.iter().enumerate() {
            let mut value = resource(&["h"]);
            value.segments[0].heading = Some("Zürich Café Café κόσμος 東京".into());
            cache
                .put_resource(
                    p,
                    id,
                    &write(
                        &value,
                        if i == 0 {
                            "quartz quartz amber नमस्ते"
                        } else {
                            "neutral"
                        },
                    ),
                )
                .unwrap();
        }
    }
    drop(cache);
    let conn = rusqlite::Connection::open(&database).unwrap();
    common::legacy::downgrade(&conn, "5");
    // An actual zero-posting document, a sparse rowid, and an orphan (discarded).
    conn.execute_batch(
        "UPDATE segments SET rowid=1001 WHERE rowid=1;
        DELETE FROM segments_fts WHERE rowid=1;
        INSERT INTO segments_fts(rowid,name,path,heading,body) VALUES(1001,'','','','');
        INSERT INTO segments_fts(rowid,body) VALUES(99999,'obsolete obsolete');
        DELETE FROM segments_fts WHERE rowid=99999;",
    )
    .unwrap();
    let before = common::legacy::postings(&conn);
    let rows = |c: &rusqlite::Connection| {
        c.prepare("SELECT rowid FROM segments ORDER BY rowid")
            .unwrap()
            .query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    let vectors = |c: &rusqlite::Connection| {
        c.prepare("SELECT hex(vector) FROM embeddings ORDER BY profile,input_hash")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    let old_rows = rows(&conn);
    let old_vectors = vectors(&conn);
    drop(SqliteCache::open(&database).unwrap());
    assert_eq!(common::legacy::postings(&conn), before);
    assert_eq!(rows(&conn), old_rows);
    assert_eq!(vectors(&conn), old_vectors);
    assert_eq!(
        conn.query_row(
            "SELECT body FROM segments_fts_p_61 WHERE rowid=1001",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        ""
    );
    for p in ["a", "東京"] {
        assert_integrity(&conn, p);
    }
    for p in ["a", "東京"] {
        assert_eq!(fts_totals(&conn, p)[0], 12);
        let table = common::legacy::table(p);
        conn.execute_batch(&format!("CREATE VIRTUAL TABLE fresh USING fts5(name,path,heading,body,tokenize='unicode61 remove_diacritics 2');
            INSERT INTO fresh(rowid,name,path,heading,body) SELECT rowid,name,path,heading,body FROM {table};")).unwrap();
        let fresh: Vec<u8> = conn
            .query_row("SELECT block FROM fresh_data WHERE id=1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts_totals(&conn, p), fresh);
        conn.execute_batch("DROP TABLE fresh").unwrap();
    }
    conn.execute_batch("VACUUM").unwrap();
    assert_eq!(rows(&conn), old_rows);
    assert_eq!(common::legacy::postings(&conn), before);
    // Reopening is idempotent; deleting the empty-token document updates N.
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.remove_resources("a", &[ids[0].clone()]).unwrap();
    assert_eq!(fts_totals(&conn, "a")[0], 11);
    assert_integrity(&conn, "a");
}

#[test]
fn migration_failure_and_unsupported_metadata_preserve_the_database() {
    for failure in [
        "future",
        "unknown",
        "missing",
        "no_meta",
        "missing_row",
        "unsorted",
    ] {
        let dir = TempDir::new().unwrap();
        let database = dir.path().join("broken.db");
        let mut cache = SqliteCache::open(&database).unwrap();
        cache.register_profile("a", &profile("a")).unwrap();
        let id: ResourceId = "mem://corpus/note".parse().unwrap();
        cache
            .put_resource("a", &id, &write(&resource(&["h"]), "amber quartz"))
            .unwrap();
        cache
            .put_embeddings("a", &[("h".into(), vec![1.0])])
            .unwrap();
        drop(cache);
        let conn = rusqlite::Connection::open(&database).unwrap();
        common::legacy::downgrade(&conn, "5");
        conn.execute_batch(match failure {
            "future" => "UPDATE meta SET value='7'",
            "unknown" => "UPDATE meta SET value='garbage'",
            "missing" => "DELETE FROM meta",
            "no_meta" => "DROP TABLE meta",
            "missing_row" => "DELETE FROM segments_fts",
            _ => "DELETE FROM segments_fts; INSERT INTO segments_fts(rowid,body) VALUES(1,'zulu alpha')",
        }).unwrap();
        let snapshot = || {
            let schema = conn
                .prepare("SELECT name,sql FROM sqlite_master ORDER BY name")
                .unwrap()
                .query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            let vector: String = conn
                .query_row("SELECT hex(vector) FROM embeddings", [], |r| r.get(0))
                .unwrap();
            (schema, vector, common::legacy::postings(&conn))
        };
        let before = snapshot();
        let error = SqliteCache::open(&database).err().expect(failure);
        if matches!(failure, "future" | "unknown") {
            let version = if failure == "future" { "7" } else { "garbage" };
            assert!(error.to_string().contains(&format!(
                "cache schema {version} is newer than this binary supports; use a newer seyk or a separate --cache-path"
            )));
        }
        assert_eq!(snapshot(), before, "{failure}");
    }
}

#[test]
fn v5_warm_migration_does_not_read_sources_or_embed() {
    let dir = TempDir::new().unwrap();
    let notes = dir.path().join("note.txt");
    fs::write(&notes, "quartz amber").unwrap();
    let database = dir.path().join("cache.db");
    let (engine, _) = shared_engine(SqliteCache::open(&database).unwrap(), Dimension::D128, "r1");
    let before = engine
        .search("alpha garden")
        .paths([&notes])
        .previews(None)
        .execute()
        .unwrap();
    drop(engine);
    let conn = rusqlite::Connection::open(&database).unwrap();
    common::legacy::downgrade(&conn, "5");
    struct NoRead;
    impl seyk_core::ResourceProvider for NoRead {
        fn list(
            &self,
            scope: &seyk_core::ResourceScope,
        ) -> seyk_core::Result<Vec<seyk_core::ProvidedResource>> {
            seyk_core::ResourceProvider::list(&seyk_core::FsProvider, scope)
        }
        fn read(&self, _: &ResourceId) -> seyk_core::Result<Vec<u8>> {
            panic!("warm migration must not read a source")
        }
        fn version(&self, id: &ResourceId) -> seyk_core::Result<ResourceVersion> {
            seyk_core::ResourceProvider::version(&seyk_core::FsProvider, id)
        }
    }
    let fake = std::sync::Arc::new(FakeEmbedder::new(Dimension::D128, "r1"));
    let engine = SearchEngine::builder()
        .shared_embedder(fake.clone())
        .provider(NoRead)
        .cache(SqliteCache::open(&database).unwrap())
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 20,
            max_chars: 100,
            overlap_chars: 0,
        }))
        .build()
        .unwrap();
    let after = engine
        .search("alpha garden")
        .paths([&notes])
        .previews(None)
        .execute()
        .unwrap();
    assert_eq!(after.hits, before.hits);
    assert_eq!(after.stats.sync.resources_updated, 0);
    assert_eq!(after.stats.sync.resources_unchanged, 1);
    assert_eq!(fake.resource_count(), 0);
}

#[test]
#[ignore = "public synthetic P11 storage, migration and churn measurement"]
fn measure_p11_storage_migration_and_churn() {
    use std::time::Instant;
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("measure.db");
    let mut cache = SqliteCache::open(&database).unwrap();
    for i in 0..32 {
        cache
            .register_profile(&format!("p{i}"), &profile("bench"))
            .unwrap();
    }
    drop(cache);
    let mut conn = rusqlite::Connection::open(&database).unwrap();
    let body = (0..200)
        .map(|i| format!("token{i:04}"))
        .collect::<Vec<_>>()
        .join(" ");
    let ids: Vec<ResourceId> = (0..20_000)
        .map(|i| format!("mem://synthetic/d{i:05}").parse().unwrap())
        .collect();
    {
        let tx = conn.transaction().unwrap();
        let table = common::legacy::table("p0");
        for (i, id) in ids.iter().enumerate() {
            let key: Vec<u8> = std::iter::once(0).chain(id.as_str().bytes()).collect();
            tx.execute("INSERT INTO resources(profile,path,modality,size_bytes,content_hash,version) VALUES('p0',?1,'text',1,'synthetic','null')",[&key]).unwrap();
            tx.execute("INSERT INTO segments(profile,path,ordinal,locator,input_hash) VALUES('p0',?1,0,'{\"kind\":\"whole_resource\"}','h')",[&key]).unwrap();
            let rowid = tx.last_insert_rowid();
            tx.execute(&format!("INSERT INTO {table}(rowid,name,path,heading,body) VALUES(?1,'synthetic','synthetic','',?2)"),rusqlite::params![rowid,body]).unwrap();
            if i < 31 {
                let p = format!("p{}", i + 1);
                tx.execute("INSERT INTO resources(profile,path,modality,size_bytes,content_hash) VALUES(?1,?2,'text',1,'synthetic')",rusqlite::params![p,key]).unwrap();
                tx.execute("INSERT INTO segments(profile,path,ordinal,locator,input_hash) VALUES(?1,?2,0,'{\"kind\":\"whole_resource\"}','h')",rusqlite::params![p,key]).unwrap();
                let rowid = tx.last_insert_rowid();
                let t = common::legacy::table(&p);
                tx.execute(
                    &format!("INSERT INTO {t}(rowid,body) VALUES(?1,?2)"),
                    rusqlite::params![rowid, body],
                )
                .unwrap();
            }
        }
        tx.commit().unwrap();
    }
    common::legacy::downgrade(&conn, "5");
    let keyword_bytes = |c: &rusqlite::Connection| {
        c.query_row(
            "SELECT sum(pgsize) FROM dbstat WHERE name GLOB 'segments_fts*'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap()
    };
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    let before = keyword_bytes(&conn);
    let start = Instant::now();
    let mut cache = SqliteCache::open(&database).unwrap();
    let migration = start.elapsed();
    assert!(cache.resource("p0", &ids[1000]).unwrap().is_some());
    let after = keyword_bytes(&conn);
    let wal = fs::metadata(format!("{}-wal", database.display()))
        .unwrap()
        .len();
    println!(
        "P11 32 profiles / 20,031 documents / 4,046,200 postings: keyword allocated bytes {before} -> {after} ({:.3}x); migration {migration:?}; WAL bytes {wal}",
        after as f64 / before as f64
    );
    let value = resource(&["h"]);
    let mut replaces = Vec::new();
    for id in &ids[..100] {
        let start = Instant::now();
        cache.put_resource("p0", id, &write(&value, &body)).unwrap();
        replaces.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut deletes = Vec::new();
    for id in &ids[100..200] {
        let start = Instant::now();
        cache
            .remove_resources("p0", std::slice::from_ref(id))
            .unwrap();
        deletes.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let start = Instant::now();
    cache.remove_resources("p0", &ids[200..300]).unwrap();
    let batch = start.elapsed();
    replaces.sort_by(f64::total_cmp);
    deletes.sort_by(f64::total_cmp);
    // Compare live-corpus totals and raw BM25 to a fresh regular FTS table after 300 edits/deletes.
    let table = common::legacy::table("p0");
    conn.execute_batch(&format!("CREATE VIRTUAL TABLE fresh USING fts5(name,path,heading,body,tokenize='unicode61 remove_diacritics 2');
        INSERT INTO fresh(rowid,name,path,heading,body) SELECT rowid,name,path,heading,body FROM {table};")).unwrap();
    let fresh: Vec<u8> = conn
        .query_row("SELECT block FROM fresh_data WHERE id=1", [], |r| r.get(0))
        .unwrap();
    assert_eq!(fts_totals(&conn, "p0"), fresh);
    let score = |t: &str| {
        conn.query_row(&format!("SELECT bm25({t},10,3,4,1) FROM {t} WHERE {t} MATCH 'token0001' ORDER BY rowid LIMIT 1"),[],|r|r.get::<_,f64>(0)).unwrap()
    };
    assert_eq!(score(&table), score("fresh"));
    assert_integrity(&conn, "p0");
    println!(
        "P11 replace ms p50={:.3} p95={:.3}; delete ms p50={:.3} p95={:.3}; batch delete 100={batch:?}; totals and BM25 exact fresh parity after 300 edits/deletes",
        replaces[50], replaces[95], deletes[50], deletes[95]
    );
}

#[test]
fn v5_migration_keeps_legacy_invalidated_segments_unindexed() {
    let dir = TempDir::new().unwrap();
    let database = dir.path().join("partial.db");
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.register_profile("a", &profile("a")).unwrap();
    let id: ResourceId = "mem://corpus/stale".parse().unwrap();
    cache
        .put_resource("a", &id, &write(&resource(&["h"]), "quartz"))
        .unwrap();
    cache
        .put_embeddings("a", &[("h".into(), vec![1.0])])
        .unwrap();
    drop(cache);
    let conn = rusqlite::Connection::open(&database).unwrap();
    common::legacy::downgrade(&conn, "5");
    conn.execute_batch(
        "DELETE FROM segments_fts;
        UPDATE resources SET modified_ns=NULL,content_hash='',version='null',descriptor_hash=NULL;",
    )
    .unwrap();
    let mut cache = SqliteCache::open(&database).unwrap();
    assert_eq!(cache.vectors_for("a", &paths(&[&id])).unwrap().len(), 1);
    assert!(
        cache
            .lexical_search(
                "a",
                &seyk_core::lexical::LexicalQuery::new(["quartz"]),
                &paths(&[&id]),
                1
            )
            .unwrap()
            .is_empty()
    );
    assert_eq!(cache.prune_orphans("a").unwrap(), 0);
    cache
        .put_resource("a", &id, &write(&resource(&["h"]), "quartz"))
        .unwrap();
    assert_eq!(fts_totals(&conn, "a")[0], 1);
    assert_integrity(&conn, "a");
}

fn raw_contains(database: &Path, needle: &str) -> bool {
    [
        database.to_path_buf(),
        database.with_extension("sqlite3-wal"),
    ]
    .iter()
    .filter_map(|path| match fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("{}: {error}", path.display()),
    })
    .any(|bytes| {
        bytes
            .windows(needle.len())
            .any(|part| part == needle.as_bytes())
    })
}

#[test]
fn deleted_cache_content_is_absent_from_database_and_wal_while_open() {
    const SENTINEL: &str = "syntheticprivatequartzsentinel";
    for operation in [
        "remove", "replace", "empty", "clear", "touch", "prune", "vector",
    ] {
        let temp = TempDir::new().unwrap();
        let database = temp.path().join("cache.sqlite3");
        let mut cache = SqliteCache::open(&database).unwrap();
        cache.register_profile("p", &profile("p")).unwrap();
        let id: ResourceId = "mem://synthetic/note".parse().unwrap();
        let mut value = resource(&["h"]);
        if matches!(operation, "touch" | "empty") {
            value.metadata.media_type = Some(SENTINEL.into());
            if operation == "empty" {
                value.segments.clear();
            }
        }
        let body = if matches!(operation, "touch" | "empty" | "prune" | "vector") {
            "ordinary"
        } else {
            SENTINEL
        };
        cache.put_resource("p", &id, &write(&value, body)).unwrap();
        // Thousands of distinct postings force multiple FTS blocks, not just one tiny row.
        let other: ResourceId = "mem://synthetic/keep".parse().unwrap();
        let words = (0..3000)
            .map(|i| format!("word{i:05}"))
            .collect::<Vec<_>>()
            .join(" ");
        cache
            .put_resource("p", &other, &write(&resource(&["keep"]), &words))
            .unwrap();
        if matches!(operation, "prune" | "vector") {
            cache
                .put_embeddings("p", &[(SENTINEL.into(), vec![1.0])])
                .unwrap();
        }
        assert!(raw_contains(&database, SENTINEL), "fixture for {operation}");
        // Put old content in both the main database and retained WAL frames.
        let observer = rusqlite::Connection::open(&database).unwrap();
        observer
            .execute_batch("PRAGMA wal_checkpoint(FULL)")
            .unwrap();
        match operation {
            "remove" => cache.remove_resources("p", &[id]).unwrap(),
            "replace" | "empty" => cache
                .put_resource("p", &id, &write(&resource(&["next"]), "ordinary"))
                .unwrap(),
            "clear" => cache.clear().unwrap(),
            "touch" => {
                value.metadata.media_type = None;
                cache
                    .touch_resource("p", &id, &value.metadata, &value.version)
                    .unwrap();
            }
            "prune" => {
                cache.prune_orphans("p").unwrap();
            }
            "vector" => {
                // Sentinel in vector bytes, rather than a key that remains live.
                cache.prune_orphans("p").unwrap();
                let bytes = SENTINEL.as_bytes();
                let vector = bytes
                    .chunks(4)
                    .map(|chunk| {
                        let mut padded = [0; 4];
                        padded[..chunk.len()].copy_from_slice(chunk);
                        f32::from_le_bytes(padded)
                    })
                    .collect();
                cache.put_embeddings("p", &[("h".into(), vector)]).unwrap();
                assert!(raw_contains(&database, SENTINEL));
                cache
                    .put_embeddings("p", &[("h".into(), vec![0.0])])
                    .unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            !raw_contains(&database, SENTINEL),
            "left readable bytes after {operation}"
        );
    }
}

#[test]
fn clear_reports_a_reader_blocking_wal_truncation() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("cache.sqlite3");
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.register_profile("p", &profile("p")).unwrap();
    let reader = rusqlite::Connection::open(&database).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM profiles;")
        .unwrap();
    let error = cache.clear().unwrap_err();
    assert!(
        matches!(error, seyk_core::Error::CacheCleanupPending),
        "{error}"
    );
    reader.execute_batch("ROLLBACK").unwrap();
    cache.clear().unwrap();
    assert_eq!(
        fs::metadata(database.with_extension("sqlite3-wal"))
            .unwrap()
            .len(),
        0
    );
}

#[test]
#[ignore = "synthetic secure deletion write-cost measurement; no models or user files"]
fn measure_secure_delete_write_cost() {
    use std::time::Instant;
    // Interleave modes so other workers' load does not systematically favour one.
    for run in 0..5 {
        for mode in ["OFF", "FAST", "ON"] {
            let temp = TempDir::new().unwrap();
            let database = temp.path().join("cache.sqlite3");
            let mut cache = SqliteCache::open(&database).unwrap();
            cache.register_profile("p", &profile("p")).unwrap();
            drop(cache);
            let mut conn = rusqlite::Connection::open(&database).unwrap();
            conn.execute_batch(&format!(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA secure_delete={mode};"
            ))
            .unwrap();
            let body = (0..200)
                .map(|i| format!("syntheticword{i:04}"))
                .collect::<Vec<_>>()
                .join(" ");
            let vector = vec![0u8; 768 * 4];
            let start = Instant::now();
            let tx = conn.transaction().unwrap();
            for i in 0..5000 {
                tx.execute("INSERT INTO segments_fts_p_70(rowid,name,path,heading,body) VALUES(?1,'note','synthetic/note','heading',?2)", rusqlite::params![i+1, body]).unwrap();
                tx.execute(
                    "INSERT INTO embeddings VALUES('p',?1,?2)",
                    rusqlite::params![format!("hash{i:05}"), vector],
                )
                .unwrap();
            }
            tx.commit().unwrap();
            let insert = start.elapsed();
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
                .unwrap();
            let start = Instant::now();
            let tx = conn.transaction().unwrap();
            tx.execute_batch("DELETE FROM segments_fts_p_70 WHERE rowid <= 1000; DELETE FROM embeddings WHERE input_hash < 'hash01000';").unwrap();
            let delete_only = start.elapsed();
            tx.execute_batch("INSERT INTO segments_fts_p_70(segments_fts_p_70) VALUES('rebuild');")
                .unwrap();
            tx.commit().unwrap();
            let delete_rebuild = start.elapsed();
            let start = Instant::now();
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
                .unwrap();
            let truncate = start.elapsed();
            let start = Instant::now();
            conn.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE)")
                .unwrap();
            let compact = start.elapsed();
            let start = Instant::now();
            conn.execute_batch("DROP TABLE segments_fts_p_70; DELETE FROM embeddings; DELETE FROM profiles; VACUUM; PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
            let clear = start.elapsed();
            eprintln!(
                "SECURE_DELETE {mode} run {run}: 5000 segments x 200 tokens + 5000 768-f32 vectors insert {insert:?}; delete-only 1000 segments/vectors {delete_only:?}; delete 1000 segments/vectors + rebuild 4000 segments {delete_rebuild:?}; truncate {truncate:?}; vacuum+truncate 4000 segments/vectors {compact:?}; clear remaining 4000 segments/vectors+vacuum+truncate {clear:?}"
            );
        }
    }
}

#[test]
fn migrations_scrub_legacy_free_pages_and_source_order_postings() {
    const SENTINEL: &str = "syntheticlegacysourcetokensentinel";
    for version in ["1", "2", "3", "4", "5"] {
        let temp = TempDir::new().unwrap();
        let database = temp.path().join("cache.sqlite3");
        let mut cache = SqliteCache::open(&database).unwrap();
        cache.register_profile("p", &profile("p")).unwrap();
        let id: ResourceId = "mem://synthetic/legacy".parse().unwrap();
        cache
            .put_resource("p", &id, &write(&resource(&["h"]), "ordinary"))
            .unwrap();
        drop(cache);
        let conn = rusqlite::Connection::open(&database).unwrap();
        conn.execute_batch("PRAGMA secure_delete=OFF; PRAGMA wal_autocheckpoint=0;")
            .unwrap();
        common::legacy::downgrade(&conn, version);
        conn.execute_batch("CREATE TABLE discarded(body TEXT);")
            .unwrap();
        conn.execute("INSERT INTO discarded VALUES(?1)", [SENTINEL.repeat(1000)])
            .unwrap();
        conn.execute_batch("DROP TABLE discarded;").unwrap();
        if version == "2" {
            conn.execute(
                "INSERT INTO segments_fts(rowid,body) VALUES(999,?1)",
                [SENTINEL],
            )
            .unwrap();
        }
        conn.execute_batch("PRAGMA wal_checkpoint(FULL)").unwrap();
        assert!(raw_contains(&database, SENTINEL));
        let cache = SqliteCache::open(&database).unwrap();
        assert!(!raw_contains(&database, SENTINEL), "version {version}");
        assert_eq!(cache.stats().unwrap().profiles[0].segments, 1);
    }
}

fn synthetic_sync_engine(database: &Path) -> SearchEngine {
    SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "scrub-sync"))
        .cache(SqliteCache::open(database).unwrap())
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 4096,
            max_chars: 8192,
            overlap_chars: 0,
        }))
        .build()
        .unwrap()
}

#[test]
fn one_rebuild_per_sync_for_any_number_of_changed_and_deleted_resources() {
    for count in [1, 7, 50] {
        let temp = TempDir::new().unwrap();
        let notes = temp.path().join("notes");
        fs::create_dir(&notes).unwrap();
        let database = temp.path().join("cache.sqlite3");
        for i in 0..count + 1 {
            fs::write(
                notes.join(format!("note{i}.txt")),
                "syntheticquartzsentinel old",
            )
            .unwrap();
        }
        let engine = synthetic_sync_engine(&database);
        let initial = engine.index().paths([&notes]).execute().unwrap();
        assert!(initial.sync.cache_cleanup.is_empty());
        for i in 0..count {
            fs::write(
                notes.join(format!("note{i}.txt")),
                "new synthetic ordinary passage",
            )
            .unwrap();
        }
        fs::remove_file(notes.join(format!("note{count}.txt"))).unwrap();
        let report = engine.index().paths([&notes]).execute().unwrap();
        assert_eq!(report.sync.resources_updated, count);
        assert_eq!(report.resources_removed, 1);
        assert_eq!(report.sync.cache_cleanup.profiles_rebuilt, 1);
        assert!(!report.sync.cache_cleanup.pending);
        assert!(!raw_contains(&database, "syntheticquartzsentinel"));
        assert!(
            engine
                .index()
                .paths([&notes])
                .execute()
                .unwrap()
                .sync
                .cache_cleanup
                .is_empty()
        );
    }
}

#[test]
fn reader_does_not_fail_fresh_noop_or_destructive_sync_and_next_sync_finishes_cleanup() {
    let temp = TempDir::new().unwrap();
    let notes = temp.path().join("notes");
    fs::create_dir(&notes).unwrap();
    let database = temp.path().join("cache.sqlite3");
    let engine = synthetic_sync_engine(&database);
    let reader = rusqlite::Connection::open(&database).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM profiles;")
        .unwrap();
    fs::write(notes.join("note.txt"), "syntheticreadersentinel old").unwrap();
    let fresh = engine.index().paths([&notes]).execute().unwrap();
    assert_eq!(fresh.sync.resources_updated, 1);
    assert!(fresh.sync.cache_cleanup.is_empty());
    assert!(
        engine
            .index()
            .paths([&notes])
            .execute()
            .unwrap()
            .sync
            .cache_cleanup
            .is_empty()
    );
    fs::write(notes.join("note.txt"), "updated synthetic ordinary passage").unwrap();
    let changed = engine.index().paths([&notes]).execute().unwrap();
    assert_eq!(changed.sync.resources_updated, 1);
    assert_eq!(changed.sync.cache_cleanup.profiles_rebuilt, 1);
    assert!(changed.sync.cache_cleanup.pending);
    assert!(raw_contains(&database, "syntheticreadersentinel"));
    assert!(
        engine
            .index()
            .paths([&notes])
            .execute()
            .unwrap()
            .sync
            .cache_cleanup
            .pending
    );
    reader.execute_batch("ROLLBACK").unwrap();
    let retry = engine.index().paths([&notes]).execute().unwrap();
    assert_eq!(retry.sync.resources_unchanged, 1);
    assert_eq!(retry.sync.cache_cleanup.profiles_rebuilt, 0);
    assert!(!retry.sync.cache_cleanup.pending);
    assert!(!raw_contains(&database, "syntheticreadersentinel"));
}

#[test]
fn noop_writes_with_a_reader_do_not_checkpoint_or_request_cleanup() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("cache.sqlite3");
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.register_profile("p", &profile("p")).unwrap();
    let id: ResourceId = "mem://synthetic/noop".parse().unwrap();
    let value = resource(&["h"]);
    cache
        .put_resource("p", &id, &write(&value, "ordinary"))
        .unwrap();
    cache
        .put_embeddings("p", &[("h".into(), vec![1.0])])
        .unwrap();
    let reader = rusqlite::Connection::open(&database).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM profiles;")
        .unwrap();
    let wal = database.with_extension("sqlite3-wal");
    let before = fs::metadata(&wal).unwrap().len();
    cache
        .touch_resource("p", &id, &value.metadata, &value.version)
        .unwrap();
    cache
        .put_embeddings("p", &[("h".into(), vec![1.0])])
        .unwrap();
    cache
        .remove_resources("p", &["mem://synthetic/missing".parse().unwrap()])
        .unwrap();
    assert_eq!(cache.prune_orphans("p").unwrap(), 0);
    assert_eq!(fs::metadata(&wal).unwrap().len(), before);
    assert!(!cache.stats().unwrap().cleanup_pending);
    assert!(cache.end_sync().unwrap().is_empty());
}

#[test]
fn migration_compaction_pending_is_retried_after_reader_closes() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("cache.sqlite3");
    let mut cache = SqliteCache::open(&database).unwrap();
    cache.register_profile("p", &profile("p")).unwrap();
    drop(cache);
    let reader = rusqlite::Connection::open(&database).unwrap();
    common::legacy::downgrade(&reader, "5");
    reader
        .execute_batch("PRAGMA secure_delete=OFF; CREATE TABLE discarded(body TEXT)")
        .unwrap();
    reader
        .execute(
            "INSERT INTO discarded VALUES(?1)",
            ["syntheticcompactsentinel".repeat(1000)],
        )
        .unwrap();
    reader
        .execute_batch(
            "DROP TABLE discarded; PRAGMA wal_checkpoint(FULL); BEGIN; SELECT * FROM meta;",
        )
        .unwrap();
    assert!(raw_contains(&database, "syntheticcompactsentinel"));
    let mut cache = SqliteCache::open(&database).unwrap();
    assert!(cache.stats().unwrap().cleanup_pending);
    let flag: String = reader
        .query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(flag, "5", "reader retains its old snapshot");
    assert!(cache.end_sync().unwrap().pending);
    drop(cache);
    reader.execute_batch("ROLLBACK").unwrap();
    let cache = SqliteCache::open(&database).unwrap();
    assert!(!cache.stats().unwrap().cleanup_pending);
    assert!(!raw_contains(&database, "syntheticcompactsentinel"));
    let flag: i64 = reader
        .query_row(
            "SELECT count(*) FROM meta WHERE key='compact_pending'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(flag, 0);
}
