#![cfg(unix)]
mod common;
use seyk_core::local_content::LocalContentProvider;
use seyk_core::{DiscoveryOptions, ResourceId, ResourceProvider, ResourceScope};
use std::fs;

fn fixture() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("local-content-")
        .tempdir()
        .unwrap()
}
fn scope(root: &std::path::Path) -> ResourceScope {
    ResourceScope {
        roots: vec![ResourceId::from_path(root).unwrap()],
        discovery: DiscoveryOptions::default(),
    }
}

#[test]
fn generated_content_is_not_listed_or_read_but_ordinary_targets_data_is_local() {
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap();
    for rel in [
        "targets/data/notes.rs",
        "Targets/gaze/Debug/Incremental/generated.rs",
    ] {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "public synthetic fixture").unwrap();
    }
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let listed = provider.list(&scope(&root)).unwrap();
    assert_eq!(listed.len(), 1, "generated Rust layout must be pruned");
    let safe = ResourceId::from_path(root.join("targets/data/notes.rs")).unwrap();
    assert_eq!(provider.read(&safe).unwrap(), b"public synthetic fixture");
    let generated =
        ResourceId::from_path(root.join("Targets/gaze/Debug/Incremental/generated.rs")).unwrap();
    assert!(provider.read(&generated).is_err());
    assert!(provider.open(&generated).is_err());
}

#[test]
fn fixed_policy_cannot_be_disabled_and_applies_to_explicit_roots_and_later_files() {
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    for rel in [
        "Library/private.rs",
        "CACHE/cache.json",
        "Caches/cached.rs",
        "Build/generated.rs",
        "DIST/generated.json",
        "Target/out.rs",
        "Node_Modules/pkg/data.rs",
        "__Pycache__/state.rs",
        "targets/gaze-3886/debug/incremental/generated.rs",
        "Targets/gaze/Release/Deps/data.json",
        "targets/gaze/debug/.fingerprint/data.rs",
        "targets/gaze/release/examples/data.rs",
        ".hidden/private.rs",
        "pending.ICLOUD/note.rs",
    ] {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "public forbidden fixture").unwrap();
        let id = ResourceId::from_path(&path).unwrap();
        assert!(provider.read(&id).is_err(), "read {rel}");
        assert!(provider.open(&id).is_err(), "open {rel}");
        assert!(provider.version(&id).is_err(), "version {rel}");
        assert!(
            LocalContentProvider::new(&[path.parent().unwrap().to_owned()]).is_err(),
            "root {rel}"
        );
    }
    for rel in [
        "targets/data/local.rs",
        "Targets/data/other.rs",
        "report/build.rs",
        "debug/deps/local.rs",
    ] {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "public ordinary fixture").unwrap();
        assert_eq!(
            provider
                .read(&ResourceId::from_path(path).unwrap())
                .unwrap(),
            b"public ordinary fixture"
        );
    }
    let mut requested = scope(&root);
    requested.discovery.include_hidden = true;
    requested.discovery.default_excludes = false;
    assert_eq!(provider.list(&requested).unwrap().len(), 4);
}

#[test]
fn links_and_replaced_roots_never_expand_the_pinned_scope() {
    use std::os::unix::fs::symlink;
    let tmp = fixture();
    let base = tmp.path().canonicalize().unwrap();
    let root = base.join("collection");
    let outside = base.join("other");
    fs::create_dir(&root).unwrap();
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("outside.rs"), "public other fixture").unwrap();
    fs::write(root.join("local.rs"), "public local fixture").unwrap();
    symlink(&root, base.join("alias")).unwrap();
    assert!(LocalContentProvider::new(&[base.join("alias")]).is_err());
    assert!(LocalContentProvider::new(&[base.join("alias/child")]).is_err());
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    symlink(&outside, root.join("linked")).unwrap();
    symlink(root.join("local.rs"), root.join("file-link.rs")).unwrap();
    for path in [
        root.join("linked/outside.rs"),
        root.join("file-link.rs"),
        outside.join("outside.rs"),
    ] {
        let id = ResourceId::from_path(path).unwrap();
        assert!(provider.read(&id).is_err());
        assert!(provider.open(&id).is_err());
        assert!(provider.version(&id).is_err());
    }
    assert_eq!(provider.list(&scope(&root)).unwrap().len(), 1);
    fs::rename(&root, base.join("retired")).unwrap();
    fs::create_dir(&root).unwrap();
    fs::write(root.join("local.rs"), "public replacement fixture").unwrap();
    let id = ResourceId::from_path(root.join("local.rs")).unwrap();
    assert!(provider.list(&scope(&root)).is_err());
    assert!(provider.read(&id).is_err());
    assert!(provider.open(&id).is_err());
    assert!(provider.version(&id).is_err());
}

#[test]
fn missing_or_nonregular_file_after_discovery_is_not_read() {
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap();
    let path = root.join("local.rs");
    fs::write(&path, "public fixture").unwrap();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let id = provider.list(&scope(&root)).unwrap().remove(0).id;
    fs::remove_file(&path).unwrap();
    assert!(provider.read(&id).is_err());
    assert!(provider.open(&id).is_err());
    fs::create_dir(&path).unwrap();
    assert!(provider.read(&id).is_err());
    assert!(provider.open(&id).is_err());
    assert!(provider.version(&id).is_err());
}

#[test]
fn broad_root_search_indexes_safe_siblings_and_filters_old_cached_generated_content() {
    use seyk_core::{Dimension, SearchEngine, SearchMode, SqliteCache};
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap();
    fs::write(root.join("safe.rs"), "ordinary quartz notes").unwrap();
    let generated = root.join("Targets/gaze/Debug/Deps/generated.rs");
    fs::create_dir_all(generated.parent().unwrap()).unwrap();
    fs::write(&generated, "generated quartz notes").unwrap();
    let cache = root.join("state/cache.sqlite3");
    // Seed the same cache/profile with an ordinary unscoped provider to prove
    // opt-in search cannot return a pre-existing excluded resource.
    let ordinary = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "policy-test"))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    ordinary.search("quartz").paths([&root]).execute().unwrap();
    drop(ordinary);
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "policy-test"))
        .cache(SqliteCache::open(&cache).unwrap())
        .provider(LocalContentProvider::new(std::slice::from_ref(&root)).unwrap())
        .build()
        .unwrap();
    let result = engine
        .search("quartz")
        .paths([&root])
        .mode(SearchMode::Lexical)
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(
        result.hits[0].resource.to_file_path().unwrap(),
        root.join("safe.rs")
    );
    let extracted = engine.extract(&result.hits[0]).unwrap();
    assert_eq!(extracted.text, "ordinary quartz notes");
}

#[cfg(any(feature = "pdf", feature = "audio"))]
#[test]
fn media_extraction_refuses_generated_sources_before_parsing_or_decoding() {
    use seyk_core::{Error, Modality, ResourceLocator, ResourceMetadata};
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap();
    let parent = root.join("Targets/gaze/Debug/Deps");
    fs::create_dir_all(&parent).unwrap();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let bytes = b"public invalid media that must not reach a parser";
    let metadata = |media: &str, modality| ResourceMetadata {
        media_type: Some(media.into()),
        modality,
        size_bytes: bytes.len() as u64,
        modified_ns: None,
        content_hash: blake3::hash(bytes).to_hex().to_string(),
    };
    #[cfg(feature = "pdf")]
    {
        let path = parent.join("document.pdf");
        fs::write(&path, bytes).unwrap();
        let id = ResourceId::from_path(path).unwrap();
        let locator = ResourceLocator::PageText {
            page: 1,
            byte_start: 0,
            byte_end: 1,
            text_version: "synthetic".into(),
            page_text_hash: "synthetic".into(),
        };
        assert!(
            matches!(seyk_core::extract_resource(&provider, &seyk_core::PdfTextExtractor,
            &id, &locator, &metadata("application/pdf", Modality::Document)),
            Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::PermissionDenied)
        );
    }
    #[cfg(feature = "audio")]
    {
        let path = parent.join("recording.wav");
        fs::write(&path, bytes).unwrap();
        let id = ResourceId::from_path(path).unwrap();
        let locator = ResourceLocator::Time {
            start_ms: 0,
            end_ms: 10,
        };
        assert!(
            matches!(seyk_core::audio::extract_samples(&provider, &id, &locator,
            &metadata("audio/wav", Modality::Audio), 7200),
            Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::PermissionDenied)
        );
    }
}

#[test]
fn deleted_ordinary_sqlite_resource_loses_record_vector_and_keyword_index() {
    use seyk_core::lexical::LexicalQuery;
    use seyk_core::{Dimension, SearchEngine, SqliteCache};
    use std::collections::HashSet;
    let tmp = fixture();
    let base = tmp.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    let path = root.join("note.rs");
    fs::write(&path, "public quartzdeletefixture").unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let wanted = HashSet::from([id.clone()]);
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "local-delete"))
        .cache(SqliteCache::open(base.join("state/cache.sqlite3")).unwrap())
        .provider(LocalContentProvider::new(std::slice::from_ref(&root)).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&root]).execute().unwrap();
    engine
        .with_cache(|cache| {
            assert!(cache.resource(engine.profile_id(), &id)?.is_some());
            assert_eq!(cache.vectors_for(engine.profile_id(), &wanted)?.len(), 1);
            assert_eq!(
                cache
                    .lexical_search(
                        engine.profile_id(),
                        &LexicalQuery::new(["quartzdeletefixture"]),
                        &wanted,
                        10
                    )?
                    .len(),
                1
            );
            Ok(())
        })
        .unwrap();
    fs::remove_file(path).unwrap();
    let report = engine.index().paths([&root]).execute().unwrap();
    assert_eq!(report.resources_removed, 1);
    engine
        .with_cache(|cache| {
            assert!(cache.resource(engine.profile_id(), &id)?.is_none());
            assert!(cache.resource_index(engine.profile_id())?.is_empty());
            assert!(cache.vectors_for(engine.profile_id(), &wanted)?.is_empty());
            assert!(
                cache
                    .lexical_search(
                        engine.profile_id(),
                        &LexicalQuery::new(["quartzdeletefixture"]),
                        &wanted,
                        10
                    )?
                    .is_empty()
            );
            let stats = cache.stats()?;
            assert_eq!(stats.profiles[0].resources, 0);
            assert_eq!(stats.profiles[0].segments, 0);
            assert_eq!(stats.profiles[0].embeddings, 0);
            Ok(())
        })
        .unwrap();
}

#[test]
fn sqlite_deletion_retains_policy_exclusions_request_exclusions_and_permission_uncertainty() {
    use seyk_core::lexical::LexicalQuery;
    use seyk_core::{Dimension, SearchEngine, SqliteCache};
    use std::collections::HashSet;
    use std::os::unix::fs::PermissionsExt;
    struct RestorePermissions(std::path::PathBuf, fs::Permissions);
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            fs::set_permissions(&self.0, self.1.clone()).unwrap();
        }
    }
    let tmp = fixture();
    let base = tmp.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    let files = [
        ("ordinary.rs", "ordinarydeletefixture"),
        ("Library/blocked.rs", "blockedretainfixture"),
        ("filter-held/held.rs", "excludedretainfixture"),
        ("unreadable/kept.rs", "uncertainretainfixture"),
    ];
    for (rel, text) in files {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    let cache_path = base.join("state/cache.sqlite3");
    let ordinary = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(
            Dimension::D128,
            "selective-delete",
        ))
        .cache(SqliteCache::open(&cache_path).unwrap())
        .build()
        .unwrap();
    ordinary.index().paths([&root]).execute().unwrap();
    assert_eq!(
        ordinary.with_cache(|c| c.stats()).unwrap().profiles[0].resources,
        4
    );
    drop(ordinary);
    for (rel, _) in files {
        fs::remove_file(root.join(rel)).unwrap();
    }
    let unreadable = root.join("unreadable");
    let _restore = RestorePermissions(
        unreadable.clone(),
        fs::metadata(&unreadable).unwrap().permissions(),
    );
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(
            Dimension::D128,
            "selective-delete",
        ))
        .cache(SqliteCache::open(cache_path).unwrap())
        .provider(LocalContentProvider::new(std::slice::from_ref(&root)).unwrap())
        .build()
        .unwrap();
    let mut options = DiscoveryOptions::default();
    options.exclude.push("filter-held".into());
    let report = engine
        .index()
        .paths([&root])
        .discovery(options)
        .execute()
        .unwrap();
    assert_eq!(report.resources_removed, 1);
    assert!(
        report
            .sync
            .skipped
            .iter()
            .any(|s| s.reason.contains("deletion check"))
    );
    engine
        .with_cache(|cache| {
            let stats = cache.stats()?;
            assert_eq!(stats.profiles[0].resources, 3);
            assert_eq!(stats.profiles[0].embeddings, 3);
            for (rel, word) in files.into_iter().skip(1) {
                let id = ResourceId::from_path(root.join(rel))?;
                let wanted = HashSet::from([id.clone()]);
                assert!(cache.resource(engine.profile_id(), &id)?.is_some());
                assert_eq!(cache.vectors_for(engine.profile_id(), &wanted)?.len(), 1);
                assert_eq!(
                    cache
                        .lexical_search(
                            engine.profile_id(),
                            &LexicalQuery::new([word]),
                            &wanted,
                            10
                        )?
                        .len(),
                    1
                );
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn replaced_requested_scope_retains_sqlite_entries_even_under_unchanged_configured_root() {
    use seyk_core::lexical::LexicalQuery;
    use seyk_core::{Dimension, SearchEngine, SqliteCache};
    use std::collections::HashSet;
    let tmp = fixture();
    let base = tmp.path().canonicalize().unwrap();
    let root = base.join("content");
    let requested = root.join("selected");
    fs::create_dir_all(&requested).unwrap();
    let path = requested.join("note.rs");
    fs::write(&path, "public replacedretainfixture").unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let wanted = HashSet::from([id.clone()]);
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(
            Dimension::D128,
            "scope-replacement",
        ))
        .cache(SqliteCache::open(base.join("state/cache.sqlite3")).unwrap())
        .provider(LocalContentProvider::new(std::slice::from_ref(&root)).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&requested]).execute().unwrap();
    fs::rename(&requested, root.join("retired")).unwrap();
    fs::create_dir(&requested).unwrap();
    let report = engine.index().paths([&requested]).execute().unwrap();
    assert_eq!(report.resources_removed, 0);
    assert!(
        report
            .sync
            .skipped
            .iter()
            .any(|s| s.reason.contains("deletion scope"))
    );
    engine
        .with_cache(|cache| {
            assert!(cache.resource(engine.profile_id(), &id)?.is_some());
            assert_eq!(cache.vectors_for(engine.profile_id(), &wanted)?.len(), 1);
            assert_eq!(
                cache
                    .lexical_search(
                        engine.profile_id(),
                        &LexicalQuery::new(["replacedretainfixture"]),
                        &wanted,
                        10
                    )?
                    .len(),
                1
            );
            Ok(())
        })
        .unwrap();
}

#[test]
fn verified_deletion_is_confined_and_missing_parent_is_ordinary_deletion() {
    use std::os::unix::fs::symlink;
    let tmp = fixture();
    let root = tmp.path().canonicalize().unwrap().join("collection");
    fs::create_dir(&root).unwrap();
    let selected = root.join("selected");
    fs::create_dir(&selected).unwrap();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let verified = provider.deletion_scope(&scope(&selected)).unwrap();
    let missing = ResourceId::from_path(selected.join("gone/note.rs")).unwrap();
    assert!(provider.is_deleted(&missing, &verified).unwrap());
    let outside = ResourceId::from_path(root.join("outside.rs")).unwrap();
    assert!(!provider.is_deleted(&outside, &verified).unwrap());
    let excluded = ResourceId::from_path(selected.join("Library/gone.rs")).unwrap();
    assert!(!provider.is_deleted(&excluded, &verified).unwrap());
    let mut included_only = verified.clone();
    included_only.discovery.include.push("*.md".into());
    assert!(!provider.is_deleted(&missing, &included_only).unwrap());
    symlink(root.join("nonexistent"), selected.join("linked")).unwrap();
    let linked = ResourceId::from_path(selected.join("linked/note.rs")).unwrap();
    assert!(provider.is_deleted(&linked, &verified).is_err());
    fs::rename(&selected, root.join("retired")).unwrap();
    fs::create_dir(&selected).unwrap();
    assert!(provider.is_deleted(&missing, &verified).is_err());
    // Replacing a configured root also cannot turn a missing child into deletion.
    let entire = provider.deletion_scope(&scope(&root)).unwrap();
    fs::rename(&root, root.with_extension("retired")).unwrap();
    fs::create_dir(&root).unwrap();
    let replaced = ResourceId::from_path(root.join("missing.rs")).unwrap();
    assert!(provider.is_deleted(&replaced, &entire).is_err());
    fs::remove_dir(&root).unwrap();
    fs::rename(root.with_extension("retired"), &root).unwrap();
}

#[test]
fn parent_deletion_scope_cannot_bypass_replaced_overlapping_configured_root() {
    let tmp = fixture();
    let parent = tmp.path().canonicalize().unwrap();
    let child = parent.join("sub");
    fs::create_dir(&child).unwrap();
    let provider = LocalContentProvider::new(&[parent.clone(), child.clone()]).unwrap();
    let verified = provider.deletion_scope(&scope(&parent)).unwrap();
    let missing = ResourceId::from_path(child.join("missing.rs")).unwrap();
    assert!(provider.is_deleted(&missing, &verified).unwrap());
    fs::rename(&child, parent.join("retired")).unwrap();
    fs::create_dir(&child).unwrap();
    assert!(provider.read(&missing).is_err());
    assert!(
        provider.is_deleted(&missing, &verified).is_err(),
        "the matched configured child root must still be verified"
    );
}
