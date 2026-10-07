mod common;

use common::FakeEmbedder;
use seyk::{
    Dimension, Error, ExtractedContent, Extractor, ResourceId, ResourceLocator, ResourceMetadata,
    Result, SearchEngine, TextExtractor,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tempfile::TempDir;

struct Custom {
    calls: Arc<AtomicUsize>,
    fail: bool,
}
impl Extractor for Custom {
    fn supports(&self, _: &ResourceId, locator: &ResourceLocator, _: &ResourceMetadata) -> bool {
        matches!(locator, ResourceLocator::Text { .. })
    }
    fn extract(
        &self,
        bytes: &[u8],
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> Result<ExtractedContent> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(blake3::hash(bytes).to_hex().as_str(), metadata.content_hash);
        if self.fail {
            return Err(Error::Config("custom extractor failed".into()));
        }
        TextExtractor.extract(bytes, source, locator, metadata)
    }
}

#[test]
fn exact_search_ranges_and_stale_files() {
    let temp = TempDir::new().unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "extract"))
        .build()
        .unwrap();
    for (name, text) in [
        (
            "note.md",
            "# Alpha\n\n**café** [文書](https://example.invalid) 🙂\n",
        ),
        ("note.txt", "alpha  café\r\n\t文書 🙂\r\n"),
        ("note.rs", "fn alpha() {\n    // 文書 🙂\n}\n"),
    ] {
        let path = temp.path().join(name);
        std::fs::write(&path, text).unwrap();
        let results = engine
            .search("alpha")
            .paths([&path])
            .previews(None)
            .execute()
            .unwrap();
        assert!(!results.hits.is_empty());
        for hit in &results.hits {
            let ResourceLocator::Text {
                byte_start,
                byte_end,
                ..
            } = hit.locator
            else {
                panic!("text locator")
            };
            assert_eq!(
                engine.extract(hit).unwrap().text.as_bytes(),
                &text.as_bytes()[byte_start as usize..byte_end as usize]
            );
        }
        std::fs::write(
            &path,
            text.replace("café", "cafe")
                .replace("alpha", "bravo")
                .replace("Alpha", "Bravo"),
        )
        .unwrap();
        assert!(matches!(
            engine.extract(&results.hits[0]),
            Err(Error::StaleResource { .. })
        ));
    }
}

#[test]
fn registry_order_hash_gate_and_failure_do_not_fall_through() {
    for fail in [false, true] {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("note.txt");
        std::fs::write(&path, "alpha source").unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let later = Arc::new(AtomicUsize::new(0));
        let engine = SearchEngine::builder()
            .embedder(FakeEmbedder::new(Dimension::D128, "registry"))
            .extractor(Custom {
                calls: calls.clone(),
                fail,
            })
            .extractor(Custom {
                calls: later.clone(),
                fail: false,
            })
            .build()
            .unwrap();
        let results = engine.search("alpha").paths([&path]).execute().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0, "search must not extract");
        let hit = &results.hits[0];
        let extracted = engine.extract(hit);
        if fail {
            assert!(matches!(extracted, Err(Error::Config(_))));
        } else {
            assert_eq!(extracted.unwrap().text, "alpha source");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(later.load(Ordering::SeqCst), 0);
        std::fs::write(path, "bravo source").unwrap();
        assert!(matches!(
            engine.extract(hit),
            Err(Error::StaleResource { .. })
        ));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "stale bytes must not reach extractor"
        );
    }
}
