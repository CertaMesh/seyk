mod common;

use common::FakeEmbedder;
use seyk_core::discovery::{ResourceKind, discover};
use seyk_core::segment::TextFormat;
use seyk_core::{
    Dimension, DiscoveryOptions, Error, FsProvider, MemoryCache, ProvidedResource, ResourceId,
    ResourceProvider, ResourceScope, ResourceVersion, Result, SearchEngine, SearchMode,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Item {
    bytes: Vec<u8>,
    revision: usize,
    title: String,
    kind: ResourceKind,
}
#[derive(Default)]
struct Notes {
    items: Mutex<BTreeMap<ResourceId, Item>>,
    reads: AtomicUsize,
    version_checks: AtomicUsize,
    unknown: AtomicBool,
    change_on_read: AtomicBool,
    deletion_checks: AtomicUsize,
    deletion_error: AtomicBool,
    scope_checks: AtomicUsize,
    scope_error: AtomicBool,
}
impl Notes {
    fn set(&self, uri: &str, text: &str) {
        let mut items = self.items.lock().unwrap();
        let id = uri.parse().unwrap();
        let revision = items.get(&id).map_or(1, |v| v.revision + 1);
        items.insert(
            id,
            Item {
                bytes: text.as_bytes().to_vec(),
                revision,
                title: "notes".into(),
                kind: ResourceKind::Text(TextFormat::Plain),
            },
        );
    }
    fn token(&self, item: &Item) -> ResourceVersion {
        if self.unknown.load(Ordering::SeqCst) {
            ResourceVersion::unknown()
        } else {
            ResourceVersion::known(format!("revision:{}", item.revision))
        }
    }
}
fn in_scope(id: &ResourceId, scope: &ResourceScope) -> bool {
    scope
        .roots
        .iter()
        .any(|r| id.as_str().starts_with(r.as_str()))
}
impl ResourceProvider for Notes {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        Ok(self
            .items
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| in_scope(id, scope))
            .map(|(id, item)| ProvidedResource {
                id: id.clone(),
                kind: item.kind,
                size_bytes: item.bytes.len() as u64,
                modified_ns: None,
                version: self.token(item),
                title: Some(item.title.clone()),
            })
            .collect())
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        let mut items = self.items.lock().unwrap();
        let item = items
            .get_mut(id)
            .ok_or_else(|| Error::Config("missing note".into()))?;
        let bytes = item.bytes.clone();
        if self.change_on_read.swap(false, Ordering::SeqCst) {
            item.revision += 1;
        }
        Ok(bytes)
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        self.version_checks.fetch_add(1, Ordering::SeqCst);
        self.items
            .lock()
            .unwrap()
            .get(id)
            .map(|item| self.token(item))
            .ok_or_else(|| Error::Config("missing note".into()))
    }
    fn deletion_scope(&self, scope: &ResourceScope) -> Result<ResourceScope> {
        self.scope_checks.fetch_add(1, Ordering::SeqCst);
        if self.scope_error.load(Ordering::SeqCst) {
            Err(Error::Config("synthetic scope failure".into()))
        } else {
            Ok(scope.clone())
        }
    }
    fn is_deleted(&self, id: &ResourceId, scope: &ResourceScope) -> Result<bool> {
        self.deletion_checks.fetch_add(1, Ordering::SeqCst);
        assert!(in_scope(id, scope), "outside-scope id reached provider");
        if self.deletion_error.load(Ordering::SeqCst) {
            return Err(Error::Config("synthetic deletion failure".into()));
        }
        Ok(in_scope(id, scope) && !self.items.lock().unwrap().contains_key(id))
    }
}
fn roots() -> [ResourceId; 1] {
    ["mem://book/".parse().unwrap()]
}
fn engine(notes: Arc<Notes>) -> (SearchEngine, Arc<FakeEmbedder>) {
    let fake = Arc::new(FakeEmbedder::new(Dimension::D128, "provider-test"));
    let engine = SearchEngine::builder()
        .shared_embedder(fake.clone())
        .shared_provider(notes)
        .cache(MemoryCache::new())
        .build()
        .unwrap();
    (engine, fake)
}

#[test]
fn memory_provider_drives_index_all_search_modes_previews_and_deletion() {
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/garden.txt", "alpha garden grows basil");
    let (engine, fake) = engine(notes.clone());
    let indexed = engine.index().roots(roots()).execute().unwrap();
    assert_eq!(indexed.sync.resources_updated, 1);
    assert!(indexed.sync.segments_embedded > 0);
    fake.reset_resource_count();
    for mode in [
        SearchMode::Semantic,
        SearchMode::Lexical,
        SearchMode::Hybrid,
    ] {
        let r = engine
            .search("alpha garden")
            .roots(roots())
            .mode(mode)
            .execute()
            .unwrap();
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].resource.as_str(), "mem://book/garden.txt");
        assert_eq!(
            r.hits[0].preview.as_deref(),
            Some("alpha garden grows basil")
        );
        assert_eq!(r.stats.sync.resources_unchanged, 1);
    }
    let r = engine
        .search("garden.txt")
        .roots(roots())
        .mode(SearchMode::Lexical)
        .execute()
        .unwrap();
    assert!(r.hits[0].signals.filename);
    assert_eq!(fake.resource_count(), 0);
    notes.reads.store(0, Ordering::SeqCst);
    engine
        .search("alpha garden")
        .roots(roots())
        .previews(None)
        .execute()
        .unwrap();
    assert_eq!(notes.reads.load(Ordering::SeqCst), 0);
    notes.set("mem://book/garden.txt", "quartz pebble in the garden");
    assert_eq!(
        engine
            .index()
            .roots(roots())
            .execute()
            .unwrap()
            .sync
            .resources_updated,
        1
    );
    assert!(fake.resource_count() > 0);
    notes.items.lock().unwrap().clear();
    assert_eq!(
        engine
            .index()
            .roots(roots())
            .execute()
            .unwrap()
            .resources_removed,
        1
    );
    assert!(
        engine
            .search("garden")
            .roots(roots())
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn unknown_versions_rehash_without_reembedding_and_stale_previews_are_hidden() {
    let notes = Arc::new(Notes::default());
    notes.unknown.store(true, Ordering::SeqCst);
    notes.set("mem://book/garden.txt", "alpha garden grows basil");
    let (engine, fake) = engine(notes.clone());
    engine.index().roots(roots()).execute().unwrap();
    fake.reset_resource_count();
    notes.reads.store(0, Ordering::SeqCst);
    assert_eq!(
        engine
            .index()
            .roots(roots())
            .execute()
            .unwrap()
            .sync
            .resources_unchanged,
        1
    );
    assert_eq!(notes.reads.load(Ordering::SeqCst), 1);
    assert_eq!(fake.resource_count(), 0);
    notes.set(
        "mem://book/garden.txt",
        "changed text that invalidates the old byte range",
    );
    let r = engine
        .search("garden")
        .roots(roots())
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(r.hits.len(), 1);
    assert!(r.hits[0].preview.is_none());
}

#[test]
fn version_race_is_skipped_and_deletion_stays_inside_scope() {
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/a.txt", "alpha garden");
    notes.set("mem://other/b.txt", "quartz pebble");
    notes.change_on_read.store(true, Ordering::SeqCst);
    let (engine, _) = engine(notes.clone());
    let report = engine.index().roots(roots()).execute().unwrap();
    assert_eq!(report.sync.skipped.len(), 1);
    assert!(report.sync.skipped[0].path.is_none());
    assert!(
        serde_json::to_value(&report.sync.skipped[0])
            .unwrap()
            .get("path")
            .is_none()
    );
    assert_eq!(report.sync.resources_updated, 0);
    engine
        .index()
        .roots([roots()[0].clone(), "mem://other/".parse().unwrap()])
        .execute()
        .unwrap();
    notes.items.lock().unwrap().clear();
    assert_eq!(
        engine
            .index()
            .roots(roots())
            .execute()
            .unwrap()
            .resources_removed,
        1
    );
    assert_eq!(
        engine
            .with_cache(|c| Ok(c.resource_index(engine.profile_id())?.len()))
            .unwrap(),
        1
    );
}

#[test]
fn filesystem_listing_preserves_discovery_and_deletion_is_conservative() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("note.md"), "alpha garden").unwrap();
    std::fs::write(temp.path().join(".hidden.txt"), "hidden").unwrap();
    let scope = ResourceScope {
        roots: vec![ResourceId::from_path(temp.path()).unwrap()],
        discovery: DiscoveryOptions::default(),
    };
    let scope = FsProvider.deletion_scope(&scope).unwrap();
    let old = discover(&[temp.path().to_path_buf()], &scope.discovery).unwrap();
    let listed = FsProvider.list(&scope).unwrap();
    assert_eq!(listed.len(), old.len());
    assert_eq!(listed[0].id, ResourceId::from_path(&old[0].path).unwrap());
    assert_eq!(FsProvider.read(&listed[0].id).unwrap(), b"alpha garden");
    assert!(
        listed[0]
            .version
            .matches(&FsProvider.version(&listed[0].id).unwrap())
    );
    let mut excluded = scope.clone();
    excluded.discovery.exclude = vec!["*.md".into()];
    assert!(FsProvider.list(&excluded).unwrap().is_empty());
    assert!(!FsProvider.is_deleted(&listed[0].id, &excluded).unwrap());
    std::fs::remove_file(listed[0].id.to_file_path().unwrap()).unwrap();
    assert!(FsProvider.is_deleted(&listed[0].id, &scope).unwrap());
    assert!(
        !FsProvider
            .is_deleted(
                &ResourceId::from_path(temp.path().parent().unwrap().join("not-in-this-scope.md"))
                    .unwrap(),
                &scope
            )
            .unwrap()
    );
    assert!(FsProvider.read(&"mem://book/a".parse().unwrap()).is_err());
}

#[cfg(unix)]
#[test]
fn path_convenience_resolves_symlinks_before_uri_dot_normalization() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join("real/sub")).unwrap();
    std::os::unix::fs::symlink(temp.path().join("real/sub"), temp.path().join("link")).unwrap();
    std::fs::write(temp.path().join("real/target.txt"), "alpha garden").unwrap();
    let fake = FakeEmbedder::new(Dimension::D128, "symlink");
    let engine = SearchEngine::builder().embedder(fake).build().unwrap();
    let results = engine
        .search("alpha garden")
        .paths([temp.path().join("link/../target.txt")])
        .execute()
        .unwrap();
    assert_eq!(
        results.hits[0].resource.to_file_path().unwrap(),
        temp.path().join("real/target.txt").canonicalize().unwrap()
    );
}

#[test]
fn descriptor_changes_match_cold_preparation_even_with_unknown_or_equal_versions() {
    for unknown in [false, true] {
        for bump in [false, true] {
            let notes = Arc::new(Notes::default());
            notes.unknown.store(unknown, Ordering::SeqCst);
            let id: ResourceId = "mem://book/note".parse().unwrap();
            notes.set(id.as_str(), "# Heading\nhello world");
            let (warm, _) = engine(notes.clone());
            warm.index().roots(roots()).execute().unwrap();
            for change_kind in [false, true] {
                {
                    let mut items = notes.items.lock().unwrap();
                    let item = items.get_mut(&id).unwrap();
                    item.title = "new title".into();
                    if change_kind {
                        item.kind = ResourceKind::Text(TextFormat::Markdown);
                    }
                    if bump {
                        item.revision += 1;
                    }
                }
                let report = warm.index().roots(roots()).execute().unwrap();
                assert_eq!(report.sync.resources_updated, 1);
                let (cold, _) = engine(notes.clone());
                cold.index().roots(roots()).execute().unwrap();
                let cached = |e: &SearchEngine| {
                    e.with_cache(|c| c.resource(e.profile_id(), &id))
                        .unwrap()
                        .unwrap()
                };
                assert_eq!(cached(&warm), cached(&cold));
            }
        }
    }
}

#[test]
fn filesystem_index_removes_deleted_resources_and_vectors() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("note.md");
    std::fs::write(&path, "alpha garden").unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "fs-delete"))
        .build()
        .unwrap();
    engine.index().paths([temp.path()]).execute().unwrap();
    std::fs::remove_file(path).unwrap();
    let report = engine.index().paths([temp.path()]).execute().unwrap();
    assert_eq!(report.resources_removed, 1);
    let stats = engine.with_cache(|c| c.stats()).unwrap();
    assert_eq!(stats.profiles[0].resources, 0);
    assert_eq!(stats.profiles[0].embeddings, 0);
}

#[test]
fn deletion_errors_retain_entries_and_do_not_abort_the_pass() {
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/a", "alpha garden");
    notes.set("mem://bookish/b", "quartz pebble");
    let (engine, _) = engine(notes.clone());
    engine
        .index()
        .roots([roots()[0].clone(), "mem://bookish/".parse().unwrap()])
        .execute()
        .unwrap();
    notes.items.lock().unwrap().clear();
    notes.deletion_checks.store(0, Ordering::SeqCst);
    notes.scope_checks.store(0, Ordering::SeqCst);
    notes.deletion_error.store(true, Ordering::SeqCst);
    let report = engine.index().roots(roots()).execute().unwrap();
    assert_eq!(notes.deletion_checks.load(Ordering::SeqCst), 1);
    assert_eq!(notes.scope_checks.load(Ordering::SeqCst), 1);
    assert_eq!(report.resources_removed, 0);
    assert_eq!(report.sync.skipped.len(), 1);
    assert!(
        report.sync.skipped[0]
            .reason
            .contains("synthetic deletion failure")
    );
    assert_eq!(
        engine
            .with_cache(|c| Ok(c.resource_index(engine.profile_id())?.len()))
            .unwrap(),
        2
    );
}

#[cfg(unix)]
#[test]
fn permission_denied_parent_retains_cache_and_index_succeeds() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let parent = temp.path().join("private");
    std::fs::create_dir(&parent).unwrap();
    let path = parent.join("note.md");
    std::fs::write(&path, "alpha garden").unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "permissions"))
        .build()
        .unwrap();
    engine.index().paths([temp.path()]).execute().unwrap();
    let permissions = std::fs::metadata(&parent).unwrap().permissions();
    std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o0)).unwrap();
    let denied =
        std::fs::metadata(&path).is_err_and(|e| e.kind() == std::io::ErrorKind::PermissionDenied);
    let result = engine.index().paths([temp.path()]).execute();
    std::fs::set_permissions(&parent, permissions).unwrap();
    let report = result.unwrap();
    assert_eq!(report.resources_removed, 0);
    if denied {
        // A privileged test runner can bypass Unix mode bits.
        assert!(
            report
                .sync
                .skipped
                .iter()
                .any(|s| s.reason.contains("deletion check"))
        );
    }
    assert_eq!(
        engine
            .with_cache(|c| Ok(c.resource_index(engine.profile_id())?.len()))
            .unwrap(),
        1
    );
}

#[test]
fn deletion_scope_errors_are_reported_without_losing_entries() {
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/a", "alpha garden");
    let (engine, _) = engine(notes.clone());
    engine.index().roots(roots()).execute().unwrap();
    notes.items.lock().unwrap().clear();
    notes.scope_error.store(true, Ordering::SeqCst);
    let report = engine.index().roots(roots()).execute().unwrap();
    assert_eq!(report.resources_removed, 0);
    assert_eq!(report.sync.skipped.len(), 1);
    assert!(
        report.sync.skipped[0]
            .reason
            .contains("synthetic scope failure")
    );
    assert_eq!(
        engine
            .with_cache(|c| Ok(c.resource_index(engine.profile_id())?.len()))
            .unwrap(),
        1
    );
}

#[test]
fn fresh_listing_avoids_version_checks_but_reads_are_guarded() {
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/a", "alpha garden");
    let (engine, _) = engine(notes.clone());
    engine.index().roots(roots()).execute().unwrap();
    assert_eq!(notes.version_checks.swap(0, Ordering::SeqCst), 2);
    engine.index().roots(roots()).execute().unwrap();
    assert_eq!(notes.version_checks.load(Ordering::SeqCst), 0);
    notes.set("mem://book/a", "quartz pebble");
    engine.index().roots(roots()).execute().unwrap();
    assert_eq!(notes.version_checks.swap(0, Ordering::SeqCst), 2);
    notes.unknown.store(true, Ordering::SeqCst);
    engine.index().roots(roots()).execute().unwrap();
    assert_eq!(notes.version_checks.load(Ordering::SeqCst), 2);
}

#[test]
fn memory_extraction_verifies_current_bytes_without_model_or_filesystem() {
    let notes = Arc::new(Notes::default());
    let text = "alpha café\r\n  文書 🙂\n";
    notes.set("mem://book/entry", text);
    let (engine, fake) = engine(notes.clone());
    let results = engine
        .search("alpha")
        .roots(roots())
        .previews(None)
        .execute()
        .unwrap();
    let hit = &results.hits[0];
    assert!(hit.preview.is_none());
    notes.reads.store(0, Ordering::SeqCst);
    fake.reset_resource_count();
    let content = engine.extract(hit).unwrap();
    let seyk_core::ResourceLocator::Text {
        byte_start,
        byte_end,
        ..
    } = hit.locator
    else {
        panic!("text locator")
    };
    assert_eq!(content.text, &text[byte_start as usize..byte_end as usize]);
    assert_eq!(content.metadata, hit.metadata);
    assert_eq!(notes.reads.load(Ordering::SeqCst), 1);
    assert_eq!(fake.resource_count(), 0);
    let mut unsupported = hit.clone();
    unsupported.locator = seyk_core::ResourceLocator::WholeResource;
    assert!(matches!(
        engine.extract(&unsupported),
        Err(Error::UnsupportedExtraction { .. })
    ));
    assert_eq!(notes.reads.load(Ordering::SeqCst), 1);
    notes.set("mem://book/entry", &text.replace("alpha", "bravo"));
    assert!(matches!(
        engine.extract(hit),
        Err(Error::StaleResource { .. })
    ));
}

#[test]
fn previews_share_one_verified_read_across_multiple_hits() {
    let notes = Arc::new(Notes::default());
    let first =
        "alpha garden with basil and tomatoes, watered in the morning before the sun gets hot.";
    let second =
        "alpha harbor with ships and a lighthouse, watched in the evening as the sun goes down.";
    notes.set("mem://book/notes.txt", &format!("{first}\n\n{second}"));
    let (engine, _) = engine(notes.clone());
    engine.index().roots(roots()).execute().unwrap();
    notes.reads.store(0, Ordering::SeqCst);
    let results = engine
        .search("alpha")
        .roots(roots())
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(results.hits.len(), 2);
    assert_eq!(notes.reads.load(Ordering::SeqCst), 1);
    for hit in &results.hits {
        assert!(matches!(hit.preview.as_deref(), Some(p) if p == first || p == second));
    }
    notes.set("mem://book/notes.txt", "changed source");
    notes.reads.store(0, Ordering::SeqCst);
    let stale = engine
        .search("alpha")
        .roots(roots())
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(stale.hits.len(), 2);
    assert!(stale.hits.iter().all(|hit| hit.preview.is_none()));
    assert_eq!(notes.reads.load(Ordering::SeqCst), 1);
    notes.reads.store(0, Ordering::SeqCst);
    engine
        .search("alpha")
        .roots(roots())
        .sync(false)
        .previews(None)
        .execute()
        .unwrap();
    assert_eq!(notes.reads.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "pdf")]
#[test]
fn opaque_pdf_provider_and_no_preview_reads() {
    #[path = "support/pdf_fixture.rs"]
    mod fixture;
    let notes = Arc::new(Notes::default());
    notes.set("mem://book/opaque", "unused");
    {
        let mut items = notes.items.lock().unwrap();
        let item = items.values_mut().next().unwrap();
        item.kind = ResourceKind::Pdf;
        item.bytes = fixture::pdf(&["garden basil", "harbor anchors"], fixture::Kind::Plain);
    }
    let (engine, _) = engine(notes.clone());
    let first = engine.search("basil").roots(roots()).execute().unwrap();
    assert!(first.hits[0].preview.as_ref().unwrap().contains("basil"));
    assert_eq!(engine.extract(&first.hits[0]).unwrap().text, "garden basil");
    let before = notes.reads.load(Ordering::SeqCst);
    let warm = engine
        .search("basil")
        .roots(roots())
        .previews(None)
        .execute()
        .unwrap();
    assert!(!warm.hits.is_empty());
    assert!(warm.hits.iter().all(|h| h.preview.is_none()));
    assert_eq!(notes.reads.load(Ordering::SeqCst), before);
}

/// The refusal must come from the placeholder check itself. Ids are built on
/// canonical paths (macOS temp directories sit behind the /var link), or the
/// link refusal would make these tests pass without that check.
fn assert_unmaterialized<T>(result: seyk_core::Result<T>, what: &str) {
    match result {
        Err(Error::Io { source, .. })
            if source.kind() == std::io::ErrorKind::PermissionDenied
                && source.to_string().contains("unmaterialized") => {}
        Err(other) => panic!("{what}: expected the placeholder refusal, got {other}"),
        Ok(_) => panic!("{what}: placeholder content was admitted"),
    }
}

#[test]
fn filesystem_placeholder_markers_refuse_both_content_read_seams() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    for suffix in ["document.txt.icloud", "document.txt.ICLOUD"] {
        let path = base.join(suffix);
        std::fs::write(&path, b"Synthetic unmaterialized-content sentinel").unwrap();
        let id = ResourceId::from_path(&path).unwrap();
        assert_unmaterialized(FsProvider.read(&id), "placeholder read");
        assert_unmaterialized(FsProvider.open(&id), "placeholder seekable open");
    }
}

#[test]
fn ordinary_local_content_remains_readable_through_both_seams() {
    use std::io::Read;
    let temp = tempfile::tempdir().unwrap();
    // Content ids are canonical; macOS temp directories sit behind /var.
    let path = temp.path().canonicalize().unwrap().join("ordinary.md");
    let bytes = b"Public synthetic ordinary local content";
    std::fs::write(&path, bytes).unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    assert_eq!(FsProvider.read(&id).unwrap(), bytes);
    let mut opened = FsProvider.open(&id).unwrap();
    let mut actual = Vec::new();
    opened.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, bytes);
}

#[cfg(unix)]
#[test]
fn symlinked_ids_fail_closed_but_explicit_symlink_roots_resolve_to_canonical_ids() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let local = base.join("local.md");
    std::fs::write(&local, b"Synthetic local alias content").unwrap();
    let alias = base.join("alias.md");
    std::os::unix::fs::symlink(&local, &alias).unwrap();
    // Reads never follow a link, even a final one; the error says why.
    let id = ResourceId::from_path(&alias).unwrap();
    let error = FsProvider.read(&id).unwrap_err().to_string();
    assert!(error.contains("symbolic link"), "{error}");
    assert!(FsProvider.open(&id).is_err());
    assert!(FsProvider.version(&id).is_err());
    // An explicit symlink root still works: discovery stores the canonical id.
    let scope = ResourceScope {
        roots: vec![ResourceId::from_path(&alias).unwrap()],
        discovery: DiscoveryOptions::default(),
    };
    let listed = FsProvider.list(&scope).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, ResourceId::from_path(&local).unwrap());
    assert_eq!(
        FsProvider.read(&listed[0].id).unwrap(),
        b"Synthetic local alias content"
    );
    assert!(FsProvider.open(&listed[0].id).is_ok());
    // A placeholder behind an alias is denied either way: the alias id as a
    // link, the canonical id it resolves to as unmaterialized content.
    std::fs::remove_file(&alias).unwrap();
    let placeholder = base.join("source.md.icloud");
    std::fs::write(&placeholder, b"Synthetic placeholder sentinel").unwrap();
    std::os::unix::fs::symlink(&placeholder, &alias).unwrap();
    let error = FsProvider.read(&id).unwrap_err().to_string();
    assert!(error.contains("symbolic link"), "{error}");
    assert!(FsProvider.open(&id).is_err());
    let canonical = ResourceId::from_path(alias.canonicalize().unwrap()).unwrap();
    assert_unmaterialized(FsProvider.read(&canonical), "placeholder read");
    assert_unmaterialized(FsProvider.open(&canonical), "placeholder seekable open");
}

#[test]
fn fresh_listing_does_not_admit_missing_or_replaced_nonregular_content() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("note.md");
    std::fs::write(&path, b"Synthetic initially local note").unwrap();
    let scope = ResourceScope {
        roots: vec![ResourceId::from_path(temp.path()).unwrap()],
        discovery: DiscoveryOptions::default(),
    };
    let listed = FsProvider.list(&scope).unwrap();
    assert_eq!(listed.len(), 1);
    let id = &listed[0].id;
    std::fs::remove_file(&path).unwrap();
    assert!(FsProvider.read(id).is_err());
    assert!(FsProvider.open(id).is_err());
    std::fs::create_dir(&path).unwrap();
    assert!(FsProvider.read(id).is_err());
    assert!(FsProvider.open(id).is_err());
}

#[test]
fn placeholder_ancestors_refuse_content_and_model_free_extraction() {
    use seyk_core::{Modality, ResourceLocator, ResourceMetadata, TextExtractor, extract_resource};
    let temp = tempfile::tempdir().unwrap();
    let folder = temp.path().canonicalize().unwrap().join("pending.icloud");
    std::fs::create_dir(&folder).unwrap();
    let path = folder.join("note.md");
    let bytes = b"Synthetic local bytes behind a placeholder marker";
    std::fs::write(&path, bytes).unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    assert_unmaterialized(FsProvider.read(&id), "placeholder ancestor read");
    assert_unmaterialized(FsProvider.open(&id), "placeholder ancestor open");
    let metadata = ResourceMetadata {
        media_type: Some("text/plain".into()),
        modality: Modality::Text,
        size_bytes: bytes.len() as u64,
        modified_ns: None,
        content_hash: blake3::hash(bytes).to_hex().to_string(),
    };
    let locator = ResourceLocator::Text {
        byte_start: 0,
        byte_end: bytes.len() as u64,
        line_start: None,
        line_end: None,
    };
    assert_unmaterialized(
        extract_resource(&FsProvider, &TextExtractor, &id, &locator, &metadata),
        "placeholder ancestor extraction",
    );
}

#[cfg(feature = "pdf")]
#[test]
fn pdf_extraction_refuses_placeholder_before_parser_dispatch() {
    use seyk_core::{
        Modality, PdfTextExtractor, ResourceLocator, ResourceMetadata, extract_resource,
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("document.pdf.icloud");
    let bytes = b"Synthetic bytes must not reach PDF parser";
    std::fs::write(&path, bytes).unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let metadata = ResourceMetadata {
        media_type: Some("application/pdf".into()),
        modality: Modality::Document,
        size_bytes: bytes.len() as u64,
        modified_ns: None,
        content_hash: blake3::hash(bytes).to_hex().to_string(),
    };
    let locator = ResourceLocator::PageText {
        page: 1,
        byte_start: 0,
        byte_end: 1,
        text_version: "synthetic".into(),
        page_text_hash: "synthetic".into(),
    };
    assert!(
        matches!(extract_resource(&FsProvider, &PdfTextExtractor, &id, &locator, &metadata), Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::PermissionDenied)
    );
}

#[cfg(feature = "audio")]
#[test]
fn audio_extraction_refuses_placeholder_before_decoder_dispatch() {
    use seyk_core::{Modality, ResourceLocator, ResourceMetadata};
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("recording.wav.icloud");
    let bytes = b"Synthetic bytes must not reach audio decoder";
    std::fs::write(&path, bytes).unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let metadata = ResourceMetadata {
        media_type: Some("audio/wav".into()),
        modality: Modality::Audio,
        size_bytes: bytes.len() as u64,
        modified_ns: None,
        content_hash: blake3::hash(bytes).to_hex().to_string(),
    };
    let locator = ResourceLocator::Time {
        start_ms: 0,
        end_ms: 10,
    };
    assert!(
        matches!(seyk_core::audio::extract_samples(&FsProvider, &id, &locator, &metadata, 7200), Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::PermissionDenied)
    );
}
