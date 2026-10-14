//! Discovery walks from the root handle it opened: an ancestor swapped for a
//! link after that point changes nothing the walk lists.
#![cfg(all(unix, debug_assertions))]

use seyk_core::DiscoveryOptions;
use seyk_core::discovery::{discover, test_hooks};
use seyk_core::local_content::LocalContentProvider;
use seyk_core::{ResourceId, ResourceProvider, ResourceScope};
use std::path::{Path, PathBuf};

fn fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let temp = tempfile::Builder::new()
        .prefix("pinned-walk-")
        .tempdir()
        .unwrap();
    let base = temp.path().canonicalize().unwrap();
    let root = base.join("a/w");
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("inside.md"), "inside").unwrap();
    std::fs::write(root.join("sub/nested.md"), "inside").unwrap();
    let outside = base.join("outside/w");
    std::fs::create_dir_all(outside.join("sub")).unwrap();
    std::fs::write(outside.join("only-outside.md"), "outside").unwrap();
    std::fs::write(outside.join("sub/outside-nested.md"), "outside").unwrap();
    (temp, base, root)
}

fn swap_on_pin(base: &Path, root: &Path) {
    let base = base.to_path_buf();
    test_hooks::after_root_pinned(root.to_path_buf(), move || {
        std::fs::rename(base.join("a"), base.join("a.real")).unwrap();
        std::os::unix::fs::symlink(base.join("outside"), base.join("a")).unwrap();
    });
}

fn restore(base: &Path) {
    std::fs::remove_file(base.join("a")).unwrap();
    std::fs::rename(base.join("a.real"), base.join("a")).unwrap();
}

#[test]
fn a_swap_after_pinning_lists_exactly_the_pinned_tree() {
    let (_temp, base, root) = fixture();
    swap_on_pin(&base, &root);
    let found = discover(std::slice::from_ref(&root), &DiscoveryOptions::default()).unwrap();
    assert!(base.join("a").is_symlink(), "hook ran");
    restore(&base);
    let found: Vec<_> = found.into_iter().map(|r| r.path).collect();
    assert_eq!(found, [root.join("inside.md"), root.join("sub/nested.md")]);
}

#[test]
fn local_content_admits_nothing_once_its_root_moved() {
    let (_temp, base, root) = fixture();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    swap_on_pin(&base, &root);
    let scope = ResourceScope {
        roots: vec![ResourceId::from_path(&root).unwrap()],
        discovery: DiscoveryOptions::default(),
    };
    let listed = provider.list(&scope).unwrap();
    assert!(base.join("a").is_symlink(), "hook ran");
    // Discovery stays on the pinned tree; every listed name is then revalidated
    // through the provider's own pinned root, which no longer resolves.
    assert!(listed.is_empty(), "{listed:?}");
    restore(&base);
    assert_eq!(provider.list(&scope).unwrap().len(), 2);
}
