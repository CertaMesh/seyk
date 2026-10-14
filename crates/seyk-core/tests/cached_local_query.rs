#![cfg(unix)]
mod common;

use seyk_core::local_content::LocalContentProvider;
use seyk_core::{
    Dimension, ProvidedResource, ResourceId, ResourceProvider, ResourceScope, ResourceVersion,
    Result, SearchEngine, SearchMode, SqliteCache,
};
use std::fs;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct ListingSpy {
    inner: LocalContentProvider,
    lists: Arc<AtomicUsize>,
    cached: Arc<AtomicUsize>,
}
impl ResourceProvider for ListingSpy {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        self.inner.list(scope)
    }
    fn supports_cached_listing(&self) -> bool {
        true
    }
    fn list_cached(
        &self,
        scope: &ResourceScope,
        ids: &[ResourceId],
    ) -> Result<Vec<ProvidedResource>> {
        self.cached.fetch_add(1, Ordering::SeqCst);
        self.inner.list_cached(scope, ids)
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        self.inner.read(id)
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        self.inner.version(id)
    }
}

#[test]
fn indexed_only_query_uses_cached_hook_and_preserves_sqlite_signals() {
    let temporary = tempfile::Builder::new()
        .prefix("cached-local-query-")
        .tempdir()
        .unwrap();
    let base = temporary.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    let path = root.join("quartz.md");
    fs::write(&path, "public quartz orchard notes").unwrap();
    let lists = Arc::new(AtomicUsize::new(0));
    let cached = Arc::new(AtomicUsize::new(0));
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "cached-query"))
        .cache(SqliteCache::open(base.join("cache.sqlite3")).unwrap())
        .provider(ListingSpy {
            inner: LocalContentProvider::new(std::slice::from_ref(&root)).unwrap(),
            lists: lists.clone(),
            cached: cached.clone(),
        })
        .build()
        .unwrap();
    engine.index().paths([&root]).execute().unwrap();
    assert_eq!(
        lists.swap(0, Ordering::SeqCst),
        1,
        "Indexing must retain complete discovery"
    );
    fs::write(root.join("uncached.md"), "public unindexed quartz orchard").unwrap();
    let response = engine
        .search("quartz")
        .paths([&root])
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(
        lists.load(Ordering::SeqCst),
        0,
        "Cached-only query must never recurse via list"
    );
    assert_eq!(cached.load(Ordering::SeqCst), 1);
    assert_eq!(
        response.stats.sync.resources_discovered, 1,
        "Only cached identities are authorized"
    );
    assert_eq!(response.hits.len(), 1);
    assert_eq!(response.hits[0].resource.to_file_path().unwrap(), path);
    assert!(
        response.hits[0].signals.semantic.is_some()
            && response.hits[0].signals.lexical.is_some()
            && response.hits[0].signals.filename
    );
    for mode in [SearchMode::Semantic, SearchMode::Lexical] {
        let result = engine
            .search("quartz")
            .paths([&root])
            .sync(false)
            .mode(mode)
            .execute()
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].resource, response.hits[0].resource);
    }
    engine
        .with_cache(|cache| {
            let id = ResourceId::from_path(&path)?;
            assert!(cache.resource(engine.profile_id(), &id)?.is_some());
            assert_eq!(
                cache
                    .vectors_for(engine.profile_id(), &std::collections::HashSet::from([id]))?
                    .len(),
                1
            );
            Ok(())
        })
        .unwrap();
    // The background indexer still discovers new files; the next query obtains
    // a fresh profile identity snapshot rather than a permanently capped set.
    engine.index().paths([&root]).execute().unwrap();
    let updated = engine
        .search("quartz")
        .paths([&root])
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(updated.stats.sync.resources_discovered, 2);
    assert_eq!(updated.hits.len(), 2);
}

#[test]
fn cached_metadata_filters_match_discovery_including_explicit_file_semantics() {
    let temporary = tempfile::Builder::new()
        .prefix("cached-policy-")
        .tempdir()
        .unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let names = [
        "safe.md",
        "nested/quartz.rs",
        "filter-held/held.md",
        "large.txt",
        "Library/blocked.rs",
        "Targets/gaze/Debug/Deps/out.rs",
        "Targets/data/local.rs",
        "pending.ICLOUD/note.md",
        "unknown.bin",
    ];
    let mut ids = Vec::new();
    for name in names {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            if name == "large.txt" {
                "public larger than limit"
            } else {
                "x"
            },
        )
        .unwrap();
        ids.push(ResourceId::from_path(path).unwrap());
    }
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let mut scope = ResourceScope {
        roots: vec![ResourceId::from_path(&root).unwrap()],
        discovery: seyk_core::DiscoveryOptions::default(),
    };
    // These all match the unsafe/oversized fixtures too, so their rejection
    // exercises fixed policy and size rather than being masked by the glob.
    scope.discovery.include = vec!["*.md".into(), "*.rs".into(), "*.txt".into()];
    scope.discovery.exclude = vec!["filter-held".into()];
    scope.discovery.max_file_bytes = 8;
    let listed = provider.list(&scope).unwrap();
    let cached = provider.list_cached(&scope, &ids).unwrap();
    assert_eq!(
        listed.iter().map(|r| &r.id).collect::<Vec<_>>(),
        cached.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
    assert_eq!(cached.len(), 3);
    assert!(
        cached
            .iter()
            .any(|r| r.id.to_file_path().unwrap() == root.join("Targets/data/local.rs"))
    );
    for (a, b) in listed.iter().zip(&cached) {
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.size_bytes, b.size_bytes);
        assert_eq!(a.modified_ns, b.modified_ns);
        assert_eq!(a.version, b.version);
        assert_eq!(a.title, b.title);
    }
    let broad = scope.roots.clone();
    scope.roots = vec![ResourceId::from_path(root.join("nested")).unwrap()];
    let narrowed = provider.list_cached(&scope, &ids).unwrap();
    assert_eq!(narrowed.len(), 1);
    assert_eq!(
        narrowed[0].id.to_file_path().unwrap(),
        root.join("nested/quartz.rs")
    );
    scope.roots = broad;
    // Explicit unknown files ignore request globs, as discover does; fixed policy still applies.
    scope.roots = vec![ids.last().unwrap().clone()];
    scope.discovery.exclude = vec!["*".into()];
    assert_eq!(provider.list_cached(&scope, &ids).unwrap().len(), 1);
    assert_eq!(provider.list(&scope).unwrap().len(), 1);
    scope.roots = vec![ResourceId::from_path(root.join("Library")).unwrap()];
    assert!(provider.list_cached(&scope, &ids).is_err());
}

#[test]
fn cached_metadata_rejects_removed_linked_nonregular_and_replaced_roots() {
    use std::os::unix::fs::symlink;
    let temporary = tempfile::Builder::new()
        .prefix("cached-current-")
        .tempdir()
        .unwrap();
    let base = temporary.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    let files = [
        "gone.md",
        "linked.md",
        "directory.md",
        "nested/note.md",
        "safe.md",
    ];
    let ids: Vec<_> = files
        .iter()
        .map(|name| {
            let path = root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "public").unwrap();
            ResourceId::from_path(path).unwrap()
        })
        .collect();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let scope = ResourceScope {
        roots: vec![ResourceId::from_path(&root).unwrap()],
        discovery: seyk_core::DiscoveryOptions::default(),
    };
    fs::remove_file(root.join("gone.md")).unwrap();
    fs::remove_file(root.join("linked.md")).unwrap();
    symlink(root.join("safe.md"), root.join("linked.md")).unwrap();
    fs::remove_file(root.join("directory.md")).unwrap();
    fs::create_dir(root.join("directory.md")).unwrap();
    fs::rename(root.join("nested"), base.join("retired-nested")).unwrap();
    symlink(base.join("retired-nested"), root.join("nested")).unwrap();
    let eligible = provider.list_cached(&scope, &ids).unwrap();
    assert_eq!(
        eligible.iter().map(|r| &r.id).collect::<Vec<_>>(),
        [&ids[4]]
    );
    fs::rename(&root, base.join("retired")).unwrap();
    fs::create_dir(&root).unwrap();
    fs::write(root.join("safe.md"), "public replacement").unwrap();
    assert!(provider.list_cached(&scope, &ids).is_err());
    assert!(
        provider.list_cached(&scope, &[]).is_err(),
        "An empty cache cannot bypass configured root identity"
    );
}

struct DefaultListingSpy {
    lists: Arc<AtomicUsize>,
}
impl ResourceProvider for DefaultListingSpy {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        seyk_core::FsProvider.list(scope)
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        seyk_core::FsProvider.read(id)
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        seyk_core::FsProvider.version(id)
    }
}

#[test]
fn default_provider_cached_only_keeps_full_listing_and_name_signals() {
    let temporary = tempfile::Builder::new()
        .prefix("cached-default-")
        .tempdir()
        .unwrap();
    let base = temporary.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("quartz.md"), "public quartz notes").unwrap();
    let lists = Arc::new(AtomicUsize::new(0));
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "default-query"))
        .cache(SqliteCache::open(base.join("cache.sqlite3")).unwrap())
        .provider(DefaultListingSpy {
            lists: lists.clone(),
        })
        .build()
        .unwrap();
    engine.index().paths([&root]).execute().unwrap();
    lists.store(0, Ordering::SeqCst);
    fs::write(root.join("uncached.md"), "public quartz notes").unwrap();
    let response = engine
        .search("quartz")
        .paths([&root])
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(lists.load(Ordering::SeqCst), 1);
    assert_eq!(response.stats.sync.resources_discovered, 2);
    assert_eq!(response.hits.len(), 1);
    assert!(response.hits[0].signals.filename && response.hits[0].signals.lexical.is_some());
    assert!(!seyk_core::FsProvider.supports_cached_listing());
}

#[test]
fn empty_index_never_lists_an_uncached_tree() {
    let temporary = tempfile::Builder::new()
        .prefix("cached-empty-")
        .tempdir()
        .unwrap();
    let base = temporary.path().canonicalize().unwrap();
    let root = base.join("content");
    fs::create_dir(&root).unwrap();
    for i in 0..100 {
        fs::write(
            root.join(format!("uncached-{i}.md")),
            "public unindexed fixture",
        )
        .unwrap();
    }
    let lists = Arc::new(AtomicUsize::new(0));
    let cached = Arc::new(AtomicUsize::new(0));
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "empty-query"))
        .cache(SqliteCache::open(base.join("cache.sqlite3")).unwrap())
        .provider(ListingSpy {
            inner: LocalContentProvider::new(std::slice::from_ref(&root)).unwrap(),
            lists: lists.clone(),
            cached: cached.clone(),
        })
        .build()
        .unwrap();
    let response = engine
        .search("public")
        .paths([&root])
        .sync(false)
        .execute()
        .unwrap();
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert_eq!(cached.load(Ordering::SeqCst), 1);
    assert_eq!(response.stats.sync.resources_discovered, 0);
    assert_eq!(response.stats.sync.segments_embedded, 0);
    assert!(response.hits.is_empty());
}
