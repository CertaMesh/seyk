mod common;

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use common::FakeEmbedder;
use gemma_search::segment::{TextSegmenter, TextSegmenterConfig};
use gemma_search::{Dimension, MemoryCache, ResourceLocator, SearchEngine, SqliteCache};
use tempfile::TempDir;

fn shared_engine(
    cache: impl gemma_search::EmbeddingCache + 'static,
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

fn search(engine: &SearchEngine, root: &Path) -> gemma_search::SearchResults {
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
