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
