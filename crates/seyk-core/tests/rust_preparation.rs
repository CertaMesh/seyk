mod common;

use common::FakeEmbedder;
use seyk_core::discovery::ResourceKind;
use seyk_core::segment::TextFormat;
use seyk_core::{
    Dimension, EmbeddingCache, Error, FsProvider, ProvidedResource, ResourceId, ResourceProvider,
    ResourceScope, ResourceVersion, SearchEngine, SearchMode, SqliteCache,
};
use std::{fs, sync::Arc};

// Reproduce legacy preparation without changing the source bytes, title,
// version token, embedding profile, or filesystem identity.
struct LegacyRust;
impl ResourceProvider for LegacyRust {
    fn list(&self, scope: &ResourceScope) -> seyk_core::Result<Vec<ProvidedResource>> {
        let mut resources = FsProvider.list(scope)?;
        for resource in &mut resources {
            if resource
                .id
                .to_file_path()
                .unwrap()
                .extension()
                .is_some_and(|ext| ext == "rs")
            {
                resource.kind = ResourceKind::Text(TextFormat::Plain);
            }
        }
        Ok(resources)
    }
    fn read(&self, id: &ResourceId) -> seyk_core::Result<Vec<u8>> {
        FsProvider.read(id)
    }
    fn version(&self, id: &ResourceId) -> seyk_core::Result<ResourceVersion> {
        FsProvider.version(id)
    }
}

const SOURCE: &str = "impl Garden {\n    fn water_orchid(&self) {\n        // orchid watering keeps the plant alive during the summer heatwave.\n    }\n\n    fn shade_orchid(&self) {\n        // orchid shade prevents wilting in a dry summer heatwave.\n    }\n}\n";

#[test]
fn old_hits_remain_valid_until_sync_then_only_rust_preparation_changes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let files = root.join("files");
    fs::create_dir(&files).unwrap();
    let rust = files.join("symbols.rs");
    fs::write(&rust, SOURCE).unwrap();
    for name in ["notes.txt", "script.py", "script.ts", "script.go"] {
        fs::write(
            files.join(name),
            "orchid prose and unsupported code stay unchanged.",
        )
        .unwrap();
    }
    let db = root.join("cache.sqlite");
    let fake = Arc::new(FakeEmbedder::new(Dimension::D128, "rust-upgrade"));
    let old = SearchEngine::builder()
        .provider(LegacyRust)
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    old.index().paths([&files]).execute().unwrap();
    let profile = old.profile_id().to_owned();
    let id = ResourceId::from_path(&rust).unwrap();
    let cache = SqliteCache::open(&db).unwrap();
    let old_rust = cache.resource(&profile, &id).unwrap().unwrap();
    assert!(old_rust.segments.iter().all(|s| s.heading.is_none()));
    let before = cache.resource_index(&profile).unwrap();
    drop(old);
    let engine = SearchEngine::builder()
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    assert_eq!(engine.profile_id(), profile);
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Semantic,
        SearchMode::Lexical,
    ] {
        let results = engine
            .search("orchid")
            .paths([&rust])
            .sync(false)
            .lexical_fallback(true)
            .previews(None)
            .mode(mode)
            .execute()
            .unwrap();
        assert!(!results.hits.is_empty());
        for hit in results.hits {
            assert!(hit.heading.is_none());
            assert!(old_rust.segments.iter().any(|s| s.locator == hit.locator));
            assert!(SOURCE.contains(&engine.extract(&hit).unwrap().text));
        }
    }
    assert!(
        !engine
            .search("symbols")
            .paths([&rust])
            .sync(false)
            .previews(None)
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
    let changed = engine.index().paths([&files]).execute().unwrap();
    assert_eq!(changed.sync.resources_updated, 1);
    assert!(changed.sync.segments_embedded > 0);
    let after = cache.resource_index(&profile).unwrap();
    for (resource, previous) in before {
        if resource != id {
            assert_eq!(after[&resource], previous);
        }
    }
    let new_rust = cache.resource(&profile, &id).unwrap().unwrap();
    assert_ne!(new_rust.descriptor_hash, old_rust.descriptor_hash);
    assert!(
        new_rust
            .segments
            .iter()
            .any(|s| s.heading.as_deref() == Some("impl Garden > fn water_orchid"))
    );
    let second = engine.index().paths([&files]).execute().unwrap();
    assert_eq!(second.sync.segments_embedded, 0);
    assert_eq!(second.sync.resources_updated, 0);

    // An absent descriptor forces preparation, but identical model inputs
    // reuse the existing vectors under the unchanged profile.
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.execute("UPDATE resources SET descriptor_hash=NULL WHERE path IN (SELECT path FROM segments WHERE heading LIKE '%water_orchid%')", []).unwrap();
    let repeated = engine.index().paths([&rust]).execute().unwrap();
    assert_eq!(repeated.sync.resources_updated, 1);
    assert_eq!(repeated.sync.segments_embedded, 0);

    let results = engine
        .search("water_orchid")
        .paths([&rust])
        .sync(false)
        .execute()
        .unwrap();
    for hit in &results.hits {
        assert!(SOURCE.contains(&engine.extract(hit).unwrap().text));
    }
    assert!(!results.hits.is_empty());
    fs::write(&rust, SOURCE.replace("summer", "winter")).unwrap();
    assert!(matches!(
        engine.extract(&results.hits[0]),
        Err(Error::StaleResource { .. })
    ));
}

#[test]
fn sqlite_heading_terms_contain_identifiers_while_evidence_keeps_full_labels() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let path = root.join("symbols.rs");
    fs::write(&path, SOURCE).unwrap();
    let db = root.join("cache.sqlite");
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "rust-fts"))
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&path]).execute().unwrap();
    let conn = rusqlite::Connection::open(&db).unwrap();
    let table = common::legacy::table(engine.profile_id());
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE temp.rust_vocab USING fts5vocab(main,{table},instance)"
    ))
    .unwrap();
    let terms: Vec<String> = conn
        .prepare("SELECT DISTINCT term FROM temp.rust_vocab WHERE col='heading' ORDER BY term")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert!(terms.contains(&"garden".into()));
    assert!(terms.contains(&"water".into()));
    for keyword in ["fn", "impl", "mod"] {
        assert!(!terms.contains(&keyword.into()));
    }
    let result = engine
        .search("water_orchid")
        .paths([&path])
        .sync(false)
        .execute()
        .unwrap();
    assert!(
        result
            .hits
            .iter()
            .any(|h| h.heading.as_deref() == Some("impl Garden > fn water_orchid"))
    );
}

#[test]
fn lexical_engine_uses_rust_symbols_and_unsupported_paths_keep_legacy_chunks() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let engine = SearchEngine::lexical().unwrap();
    for extension in ["rs", "py", "ts", "go"] {
        let path = root.join(format!("symbols.{extension}"));
        fs::write(&path, SOURCE).unwrap();
        let results = engine
            .search("water_orchid")
            .paths([path])
            .execute()
            .unwrap();
        assert!(!results.hits.is_empty());
        assert_eq!(
            results.hits.iter().any(|h| h.heading.is_some()),
            extension == "rs"
        );
    }
}

#[test]
fn failed_upgrade_keeps_old_bindings_atomic_and_retryable() {
    use seyk_core::{Embedder, Embedding, Modality, ModelIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Failing {
        inner: FakeEmbedder,
        fail: AtomicBool,
    }
    impl Embedder for Failing {
        fn identity(&self) -> &ModelIdentity {
            self.inner.identity()
        }
        fn supports(&self, modality: Modality) -> bool {
            self.inner.supports(modality)
        }
        fn embed_batch(
            &self,
            requests: &[seyk_embed::EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<Embedding>> {
            if self.fail.load(Ordering::SeqCst)
                && requests
                    .iter()
                    .any(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
            {
                return Err(seyk_embed::Error::Embedding(
                    "synthetic preparation upgrade failure".into(),
                ));
            }
            self.inner.embed_batch(requests)
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let path = root.join("symbols.rs");
    fs::write(&path, SOURCE).unwrap();
    let db = root.join("cache.sqlite");
    let fake = Arc::new(Failing {
        inner: FakeEmbedder::new(Dimension::D128, "retry-rust"),
        fail: AtomicBool::new(false),
    });
    let old = SearchEngine::builder()
        .provider(LegacyRust)
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    old.index().paths([&path]).execute().unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let cache = SqliteCache::open(&db).unwrap();
    let previous = cache.resource(old.profile_id(), &id).unwrap().unwrap();
    let engine = SearchEngine::builder()
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    fake.fail.store(true, Ordering::SeqCst);
    let failed = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(failed.sync.resources_updated, 0);
    assert!(!failed.sync.skipped.is_empty());
    assert_eq!(
        cache.resource(engine.profile_id(), &id).unwrap().unwrap(),
        previous
    );
    let hits = engine
        .search("orchid")
        .paths([&path])
        .sync(false)
        .execute()
        .unwrap()
        .hits;
    // Keep the existing failed-preparation fence for this engine lifetime.
    // Persistent bindings remain intact, and a fresh cached-only reader can
    // still use the same-version evidence until a successful replacement.
    assert!(hits.is_empty());
    let reader = SearchEngine::builder()
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    let hits = reader
        .search("orchid")
        .paths([&path])
        .sync(false)
        .execute()
        .unwrap()
        .hits;
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.heading.is_none()));
    fake.fail.store(false, Ordering::SeqCst);
    let retried = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(retried.sync.resources_updated, 1);
    assert!(
        cache
            .resource(engine.profile_id(), &id)
            .unwrap()
            .unwrap()
            .segments
            .iter()
            .any(|s| s.heading.is_some())
    );
}

#[test]
fn uncached_lexical_fallback_uses_the_same_rust_preparation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let path = root.join("symbols.rs");
    fs::write(&path, SOURCE).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "fallback-rust"))
        .build()
        .unwrap();
    let result = engine
        .search("water_orchid")
        .paths([&path])
        .sync(false)
        .lexical_fallback(true)
        .execute()
        .unwrap();
    assert!(!result.hits.is_empty());
    assert!(
        result
            .hits
            .iter()
            .any(|h| h.heading.as_deref() == Some("impl Garden > fn water_orchid"))
    );
    for hit in result.hits {
        assert!(SOURCE.contains(&engine.extract(&hit).unwrap().text));
    }
}

#[test]
fn unknown_versions_keep_cached_hits_then_reprepare_rust_during_sync() {
    struct Unknown {
        legacy: bool,
    }
    impl ResourceProvider for Unknown {
        fn list(&self, scope: &ResourceScope) -> seyk_core::Result<Vec<ProvidedResource>> {
            let mut resources = if self.legacy {
                LegacyRust.list(scope)?
            } else {
                FsProvider.list(scope)?
            };
            for resource in &mut resources {
                resource.version = ResourceVersion::unknown();
            }
            Ok(resources)
        }
        fn read(&self, id: &ResourceId) -> seyk_core::Result<Vec<u8>> {
            FsProvider.read(id)
        }
        fn version(&self, _: &ResourceId) -> seyk_core::Result<ResourceVersion> {
            Ok(ResourceVersion::unknown())
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let path = root.join("symbols.rs");
    fs::write(&path, SOURCE).unwrap();
    let db = root.join("cache.sqlite");
    let fake = Arc::new(FakeEmbedder::new(Dimension::D128, "unknown-rust"));
    let old = SearchEngine::builder()
        .provider(Unknown { legacy: true })
        .shared_embedder(fake.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    old.index().paths([&path]).execute().unwrap();
    let engine = SearchEngine::builder()
        .provider(Unknown { legacy: false })
        .shared_embedder(fake)
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    let result = engine
        .search("water_orchid")
        .paths([&path])
        .sync(false)
        .lexical_fallback(true)
        .execute()
        .unwrap();
    assert!(!result.hits.is_empty());
    assert!(result.degraded.is_none());
    assert!(result.hits.iter().all(|h| h.heading.is_none()));
    for hit in result.hits {
        assert!(SOURCE.contains(&engine.extract(&hit).unwrap().text));
    }
    let upgraded = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(upgraded.sync.resources_updated, 1);
    let result = engine
        .search("water_orchid")
        .paths([&path])
        .sync(false)
        .execute()
        .unwrap();
    assert!(
        result
            .hits
            .iter()
            .any(|h| h.heading.as_deref() == Some("impl Garden > fn water_orchid"))
    );
    assert_eq!(
        engine
            .index()
            .paths([&path])
            .execute()
            .unwrap()
            .sync
            .segments_embedded,
        0
    );
}
