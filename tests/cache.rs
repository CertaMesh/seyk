mod common;

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use common::FakeEmbedder;
use seyk::cache::{CacheProfile, CachedResource, CachedSegment};
use seyk::segment::{TextSegmenter, TextSegmenterConfig};
use seyk::{
    Dimension, EmbeddingCache, MemoryCache, Modality, ModelIdentity, ResourceLocator,
    ResourceMetadata, SearchEngine, SqliteCache,
};
use tempfile::TempDir;

fn shared_engine(
    cache: impl seyk::EmbeddingCache + 'static,
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

fn search(engine: &SearchEngine, root: &Path) -> seyk::SearchResults {
    engine
        .search("alpha garden")
        .paths([root])
        .execute()
        .unwrap()
}

#[test]
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
        second_results
            .hits
            .iter()
            .all(|hit| hit.resource.path().exists())
    );
    assert!(
        first_results
            .hits
            .iter()
            .all(|hit| hit.resource.path().exists())
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
        metadata: ResourceMetadata {
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
                lexical_text: None,
            })
            .collect(),
    }
}

fn paths(items: &[&Path]) -> HashSet<PathBuf> {
    items.iter().map(|p| p.to_path_buf()).collect()
}

fn hashes(items: &[&str]) -> Vec<String> {
    items.iter().map(|h| (*h).to_string()).collect()
}

#[test]
fn memory_cache_keeps_profiles_apart_for_the_same_path_and_input() {
    let mut cache = MemoryCache::new();
    let path = Path::new("/notes.txt");
    for (id, value) in [("a", 1.0), ("b", 2.0)] {
        cache.register_profile(id, &profile(id)).unwrap();
        cache
            .put_embeddings(id, &[("h".into(), vec![value])])
            .unwrap();
        cache.put_resource(id, path, &resource(&["h"])).unwrap();
    }

    for (id, value) in [("a", 1.0), ("b", 2.0)] {
        let found = cache.vectors_for(id, &paths(&[path])).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].vector, vec![value]);
    }

    cache.touch_resource("a", path, Some(7)).unwrap();
    cache.remove_resources("b", &[path.to_path_buf()]).unwrap();
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
    let path = Path::new("/notes.txt");
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

    cache.put_resource("p", path, &resource(&["h"])).unwrap();
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
    let (first, second) = (Path::new("/first.txt"), Path::new("/second.txt"));
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
        .put_resource("p", first, &resource(&["shared", "own"]))
        .unwrap();
    cache
        .put_resource("p", second, &resource(&["shared"]))
        .unwrap();

    let only_second = cache.vectors_for("p", &paths(&[second])).unwrap();
    assert_eq!(only_second.len(), 1);
    assert_eq!(only_second[0].path, second);

    assert_eq!(cache.prune_orphans("p").unwrap(), 1);
    // A vector still used by another resource survives its sibling's removal.
    cache.remove_resources("p", &[first.to_path_buf()]).unwrap();
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
fn lexical_ties_use_path_bytes_then_ordinal_before_limit_in_both_caches() {
    let names = [
        "/notes/zulu.md",
        "/notes/alpha.md",
        "/notes/delta.md",
        "/notes/bravo.md",
    ];
    let wanted: HashSet<PathBuf> = names.iter().map(PathBuf::from).collect();
    let query = seyk::lexical::LexicalQuery::new(["quartz"]);
    let check = |cache: &mut dyn EmbeddingCache| {
        cache.register_profile("ties", &profile("ties")).unwrap();
        for name in names {
            let mut r = resource(&["first", "second"]);
            for segment in &mut r.segments {
                segment.lexical_text = Some("quartz pebble".into());
            }
            cache.put_resource("ties", Path::new(name), &r).unwrap();
        }
        let hits = cache.lexical_search("ties", &query, &wanted, 100).unwrap();
        assert_eq!(hits.len(), 8);
        assert!(hits.iter().all(|h| h.score == hits[0].score));
        let order: Vec<_> = hits.into_iter().map(|h| (h.path, h.ordinal)).collect();
        for limit in [0, 1, 3, 8] {
            let limited: Vec<_> = cache
                .lexical_search("ties", &query, &wanted, limit)
                .unwrap()
                .into_iter()
                .map(|h| (h.path, h.ordinal))
                .collect();
            assert_eq!(limited, order[..limit]);
        }
        order
    };
    let db = TempDir::new().unwrap();
    let expected = check(&mut SqliteCache::open(db.path().join("cache.sqlite3")).unwrap());
    let mut sorted = expected.clone();
    sorted.sort_by(|a, b| {
        a.0.as_os_str()
            .as_encoded_bytes()
            .cmp(b.0.as_os_str().as_encoded_bytes())
            .then(a.1.cmp(&b.1))
    });
    assert_eq!(expected, sorted);
    for _ in 0..20 {
        assert_eq!(check(&mut MemoryCache::new()), expected);
    }
}
