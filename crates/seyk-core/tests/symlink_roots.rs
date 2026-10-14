//! Outside local-content mode, explicit links keep working: caller paths are
//! resolved through symlinks before URI normalization, so links in a root
//! yield the same canonical ids, hits and reads as the real paths.
#![cfg(unix)]
mod common;

use seyk_core::{Dimension, FsProvider, ResourceId, ResourceProvider, SearchEngine};
use std::path::{Path, PathBuf};

const REAL: &str = "quartz garden note in the real folder";
const DECOY: &str = "quartz garden decoy where a lexical .. would land";

/// Built under the uncanonicalized temp path, so on macOS every root also
/// passes through the /var alias.
struct Fixture {
    temp: tempfile::TempDir,
    /// Canonical `base/real/note.md`.
    note: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        std::fs::create_dir_all(base.join("real/sub")).unwrap();
        std::fs::write(base.join("real/note.md"), REAL).unwrap();
        // `link/../note.md` lexically names this file; through the link it
        // names real/note.md.
        std::fs::write(base.join("note.md"), DECOY).unwrap();
        std::os::unix::fs::symlink(base.join("real"), base.join("dir-link")).unwrap();
        std::os::unix::fs::symlink(base.join("real/note.md"), base.join("file-link.md")).unwrap();
        std::os::unix::fs::symlink(base.join("real/sub"), base.join("sub-link")).unwrap();
        let note = base.join("real/note.md").canonicalize().unwrap();
        Self { temp, note }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.temp.path().join(rel)
    }
}

fn engine() -> SearchEngine {
    SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "symlink-roots"))
        .build()
        .unwrap()
}

/// Index, search, read and extract through `root`; everything must name and
/// return the canonical real note.
fn assert_resolves_to_real_note(fixture: &Fixture, root: &Path) {
    let engine = engine();
    engine.index().paths([root]).execute().unwrap();
    let result = engine
        .search("quartz garden")
        .paths([root])
        .execute()
        .unwrap();
    let expected = ResourceId::from_path(&fixture.note).unwrap();
    let ids: Vec<_> = result.hits.iter().map(|h| h.resource.clone()).collect();
    assert!(!ids.is_empty(), "no hits through {}", root.display());
    assert!(ids.iter().all(|id| *id == expected), "{ids:?}");
    assert_eq!(FsProvider.read(&expected).unwrap(), REAL.as_bytes());
    assert!(FsProvider.open(&expected).is_ok());
    assert!(FsProvider.version(&expected).unwrap().is_known());
    let extracted = engine.extract(&result.hits[0]).unwrap();
    assert_eq!(extracted.source, expected);
    assert!(REAL.contains(extracted.text.trim()), "{}", extracted.text);
}

#[test]
fn a_search_root_that_links_to_a_directory() {
    let fixture = Fixture::new();
    assert_resolves_to_real_note(&fixture, &fixture.path("dir-link"));
}

#[test]
fn an_explicit_file_root_that_links_to_a_file() {
    let fixture = Fixture::new();
    assert_resolves_to_real_note(&fixture, &fixture.path("file-link.md"));
}

#[test]
fn a_parent_component_after_a_link_keeps_its_filesystem_meaning() {
    let fixture = Fixture::new();
    assert_resolves_to_real_note(&fixture, &fixture.path("sub-link/../note.md"));
}
