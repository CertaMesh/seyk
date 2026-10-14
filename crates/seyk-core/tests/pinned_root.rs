//! An in-scope ancestor swapped for a link to an outside tree must not let
//! discovery list, or a provider read, that outside content under in-scope ids.
#![cfg(unix)]
mod common;

use seyk_core::local_content::LocalContentProvider;
use seyk_core::{
    Dimension, DiscoveryOptions, FsProvider, ResourceId, ResourceProvider, ResourceScope,
    SearchEngine, SearchMode,
};
use std::io::Read;
use std::path::{Path, PathBuf};

const INSIDE: &[u8] = b"quartz garden inside the walked tree";
const OUTSIDE: &[u8] = b"zygomorphic quartz outside the walked tree";

/// `base/a/w` is the walked root; `base/outside/w` mirrors it with different
/// bytes and one extra name. `swap` turns the ancestor `a` into a link there.
struct Fixture {
    _temp: tempfile::TempDir,
    base: PathBuf,
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("pinned-root-")
            .tempdir()
            .unwrap();
        let base = temp.path().canonicalize().unwrap();
        let root = base.join("a/w");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("shared.md"), INSIDE).unwrap();
        std::fs::write(root.join("inner.md"), INSIDE).unwrap();
        let outside = base.join("outside/w");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("shared.md"), OUTSIDE).unwrap();
        std::fs::write(outside.join("only-outside.md"), OUTSIDE).unwrap();
        Self {
            _temp: temp,
            base,
            root,
        }
    }

    fn swap(&self) -> impl FnOnce() + Send + 'static {
        let base = self.base.clone();
        move || swap(&base)
    }

    fn restore(&self) {
        std::fs::remove_file(self.base.join("a")).unwrap();
        std::fs::rename(self.base.join("a.real"), self.base.join("a")).unwrap();
    }

    fn id(&self, name: &str) -> ResourceId {
        ResourceId::from_path(self.root.join(name)).unwrap()
    }

    fn scope(&self) -> ResourceScope {
        ResourceScope {
            roots: vec![ResourceId::from_path(&self.root).unwrap()],
            discovery: DiscoveryOptions::default(),
        }
    }
}

fn swap(base: &Path) {
    std::fs::rename(base.join("a"), base.join("a.real")).unwrap();
    std::os::unix::fs::symlink(base.join("outside"), base.join("a")).unwrap();
}

fn names(ids: &[ResourceId]) -> Vec<String> {
    let mut names: Vec<_> = ids
        .iter()
        .map(|id| {
            id.to_file_path()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

/// Every content seam either fails or returns the in-tree bytes.
fn assert_never_outside(provider: &dyn ResourceProvider, id: &ResourceId) {
    if let Ok(bytes) = provider.read(id) {
        assert_eq!(bytes, INSIDE, "read returned outside bytes");
    }
    if let Ok(mut file) = provider.open(id) {
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, INSIDE, "open returned outside bytes");
    }
}

#[cfg(debug_assertions)]
#[test]
fn ancestor_swapped_after_root_resolution_lists_nothing_outside() {
    let fixture = Fixture::new();
    seyk_core::discovery::test_hooks::after_root_resolved(fixture.root.clone(), fixture.swap());
    let listed = FsProvider.list(&fixture.scope());
    assert!(fixture.base.join("a").is_symlink(), "hook ran");
    if let Ok(listed) = listed {
        let ids: Vec<_> = listed.into_iter().map(|r| r.id).collect();
        assert!(
            !names(&ids).contains(&"only-outside.md".to_string()),
            "outside name listed: {:?}",
            names(&ids)
        );
    }
    for name in ["shared.md", "only-outside.md"] {
        assert_never_outside(&FsProvider, &fixture.id(name));
    }
    fixture.restore();
}

#[test]
fn reads_of_listed_ids_never_follow_a_swapped_ancestor() {
    let fixture = Fixture::new();
    let listed = FsProvider.list(&fixture.scope()).unwrap();
    let ids: Vec<_> = listed.into_iter().map(|r| r.id).collect();
    assert_eq!(names(&ids), ["inner.md", "shared.md"]);
    swap(&fixture.base);
    let shared = fixture.id("shared.md");
    assert_never_outside(&FsProvider, &shared);
    let error = FsProvider.read(&shared).unwrap_err().to_string();
    assert!(error.contains("symbolic link"), "{error}");
    assert!(FsProvider.open(&shared).is_err());
    assert!(FsProvider.version(&shared).is_err());
    fixture.restore();
    assert_eq!(FsProvider.read(&shared).unwrap(), INSIDE);
    assert!(FsProvider.version(&shared).unwrap().is_known());
}

#[cfg(debug_assertions)]
#[test]
fn search_during_a_swap_returns_no_outside_text() {
    let fixture = Fixture::new();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D256, "pinned-root"))
        .build()
        .unwrap();
    seyk_core::discovery::test_hooks::after_root_resolved(fixture.root.clone(), fixture.swap());
    // The pass may fail outright; it must not index outside bytes.
    let _ = engine.index().paths([&fixture.root]).execute();
    fixture.restore();
    let result = engine
        .search("zygomorphic")
        .paths([&fixture.root])
        .mode(SearchMode::Lexical)
        .sync(false)
        .execute()
        .unwrap();
    assert!(result.hits.is_empty(), "outside text indexed: {result:?}");
}

#[cfg(debug_assertions)]
#[test]
fn local_content_lists_and_reads_nothing_outside_during_a_swap() {
    let fixture = Fixture::new();
    let provider = LocalContentProvider::new(std::slice::from_ref(&fixture.root)).unwrap();
    seyk_core::discovery::test_hooks::after_root_resolved(fixture.root.clone(), fixture.swap());
    if let Ok(listed) = provider.list(&fixture.scope()) {
        let ids: Vec<_> = listed.into_iter().map(|r| r.id).collect();
        assert!(!names(&ids).contains(&"only-outside.md".to_string()));
    }
    for name in ["shared.md", "only-outside.md"] {
        assert_never_outside(&provider, &fixture.id(name));
    }
    fixture.restore();
    assert_eq!(provider.read(&fixture.id("shared.md")).unwrap(), INSIDE);
}
