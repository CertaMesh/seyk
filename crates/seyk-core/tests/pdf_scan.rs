#![cfg(feature = "pdf")]
mod common;
#[path = "support/scan_fixture.rs"]
mod fixture;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Modality, ModelIdentity,
    ResourceInput, ResourceLocator, SearchEngine,
};
#[cfg(feature = "pdf-render")]
use seyk_core::{Error, SearchMode, SqliteCache};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "pdf-render")]
use std::sync::Arc;

struct Model {
    identity: ModelIdentity,
    vision: bool,
    calls: Mutex<Vec<usize>>,
    malformed: AtomicUsize,
}
impl Model {
    fn new(vision: bool) -> Self {
        Self {
            identity: common::FakeEmbedder::new(Dimension::D128, "scan")
                .identity()
                .clone(),
            vision,
            calls: Mutex::new(Vec::new()),
            malformed: AtomicUsize::new(0),
        }
    }
}
impl Embedder for Model {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    fn supports(&self, m: Modality) -> bool {
        m == Modality::Text || (self.vision && m == Modality::Image)
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        let resources = requests
            .iter()
            .any(|r| r.task == EmbeddingTask::SearchDocument);
        if resources {
            self.calls.lock().unwrap().push(requests.len());
        }
        if resources
            && (self.malformed.load(Ordering::Relaxed) == 1
                || (self.malformed.load(Ordering::Relaxed) == 3
                    && self.calls.lock().unwrap().len() > 1))
        {
            return Ok(Vec::new());
        }
        requests
            .iter()
            .map(|r| {
                if resources {
                    assert_eq!(r.task, EmbeddingTask::SearchDocument);
                }
                if let ResourceInput::Image { bytes } = &r.input {
                    assert!(self.vision);
                    assert!(bytes.starts_with(b"\x89PNG"));
                }
                let dimension = if resources && self.malformed.load(Ordering::Relaxed) == 2 {
                    Dimension::D256
                } else {
                    Dimension::D128
                };
                Embedding::from_raw(vec![1.0; dimension.get()], dimension)
            })
            .collect()
    }
}

#[test]
fn vision_off_reports_pages_without_skipping_the_resource() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    std::fs::write(
        &path,
        fixture::rgb_image(2, 1, &[255, 0, 0, 0, 0, 255], true),
    )
    .unwrap();
    let engine = SearchEngine::builder()
        .embedder(Model::new(false))
        .build()
        .unwrap();
    let results = engine.search("basil").paths([&path]).execute().unwrap();
    assert!(results.stats.sync.skipped.is_empty());
    assert_eq!(results.stats.sync.page_diagnostics.len(), 1);
    assert_eq!(results.stats.sync.page_diagnostics[0].page, 2);
    assert_eq!(results.hits.len(), 1);
    assert!(matches!(
        results.hits[0].locator,
        ResourceLocator::PageText { page: 1, .. }
    ));
    // A scan-only resource is still a page diagnostic, even without pdf-render.
    std::fs::write(
        &path,
        fixture::rgb_image(2, 1, &[255, 0, 0, 0, 0, 255], false),
    )
    .unwrap();
    let report = engine.index().paths([&path]).execute().unwrap();
    assert!(report.sync.skipped.is_empty());
    assert_eq!(report.sync.page_diagnostics.len(), 1);
}

#[cfg(feature = "pdf-render")]
#[test]
fn mixed_hits_have_correct_locators_no_scan_preview_and_no_scan_text() {
    for sqlite in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orchard.pdf");
        std::fs::write(
            &path,
            fixture::rgb_image(2, 1, &[255, 0, 0, 0, 0, 255], true),
        )
        .unwrap();
        let mut builder = SearchEngine::builder().embedder(Model::new(true));
        if sqlite {
            builder = builder.cache(SqliteCache::open(dir.path().join("cache.db")).unwrap());
        }
        let engine = builder.build().unwrap();
        let result = engine
            .search("orchard")
            .paths([&path])
            .mode(SearchMode::Semantic)
            .execute()
            .unwrap();
        assert!(result.stats.sync.skipped.is_empty());
        assert!(result.stats.sync.page_diagnostics.is_empty());
        assert_eq!(result.hits.len(), 2);
        let scan = result
            .hits
            .iter()
            .find(|h| matches!(h.locator, ResourceLocator::Page { page: 2 }))
            .unwrap();
        assert!(scan.preview.is_none());
        assert!(matches!(
            engine.extract(scan),
            Err(Error::PdfOcrUnavailable { page: 2, .. })
        ));
        let text = result
            .hits
            .iter()
            .find(|h| matches!(h.locator, ResourceLocator::PageText { page: 1, .. }))
            .unwrap();
        assert!(text.preview.as_ref().unwrap().contains("basil"));
        assert_eq!(
            engine
                .search("basil")
                .paths([&path])
                .mode(SearchMode::Lexical)
                .execute()
                .unwrap()
                .hits
                .len(),
            1
        );
        assert_eq!(
            engine
                .search("orchard.pdf")
                .paths([&path])
                .mode(SearchMode::Lexical)
                .execute()
                .unwrap()
                .hits
                .len(),
            2
        );
        let warm = engine.index().paths([&path]).execute().unwrap();
        assert_eq!(warm.sync.resources_unchanged, 1);
        assert_eq!(warm.sync.segments_embedded, 0);
    }
}

#[cfg(feature = "pdf-render")]
#[test]
fn page_batches_are_bounded_and_malformed_vectors_never_commit_a_resource() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drawings.pdf");
    let contents: Vec<_> = (0..9)
        .map(|i| format!("1 0 0 rg {} 100 80 120 re f", 40 + i * 10))
        .collect();
    let pages: Vec<_> = contents.iter().map(|s| (s.as_bytes(), "")).collect();
    let bytes = fixture::document(&pages, None);
    std::fs::write(&path, &bytes).unwrap();
    let model = Arc::new(Model::new(true));
    let engine = SearchEngine::builder()
        .shared_embedder(model.clone())
        .batch_size(100)
        .build()
        .unwrap();
    let report = engine.index().paths([&path]).execute().unwrap();
    assert!(report.sync.skipped.is_empty());
    assert_eq!(report.sync.segments_embedded, 9);
    assert_eq!(*model.calls.lock().unwrap(), vec![4, 4, 1]);
    let resource = seyk_core::ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
    let assert_old_binding_retained = || {
        let cached = engine
            .with_cache(|c| c.resource(engine.profile_id(), &resource))
            .unwrap()
            .unwrap();
        assert_eq!(cached.segments.len(), 9);
        assert_eq!(
            cached.metadata.content_hash,
            blake3::hash(&bytes).to_hex().as_str()
        );
    };
    for malformed in [1, 2] {
        model.malformed.store(malformed, Ordering::Relaxed);
        std::fs::write(
            &path,
            fixture::drawing(&format!(
                "/Rotate {}",
                if malformed == 1 { 90 } else { 180 }
            )),
        )
        .unwrap();
        let report = engine.index().paths([&path]).execute().unwrap();
        assert_eq!(report.sync.skipped.len(), 1);
        assert_eq!(report.sync.resources_updated, 0);
        assert_eq!(report.sync.segments_embedded, 0);
        let old = engine
            .search("anything")
            .paths([&path])
            .sync(false)
            .previews(None)
            .limit(20)
            .execute()
            .unwrap();
        // Failed current versions retain their last committed binding for
        // recovery, but must not expose it as current search evidence.
        assert!(old.hits.is_empty());
        assert_old_binding_retained();
    }
    // A later failed batch must not commit the already embedded prefix.
    let changed: Vec<_> = (0..9)
        .map(|i| format!("0 0 1 rg {} 200 80 120 re f", 140 + i * 10))
        .collect();
    let pages: Vec<_> = changed.iter().map(|s| (s.as_bytes(), "")).collect();
    std::fs::write(&path, fixture::document(&pages, None)).unwrap();
    model.calls.lock().unwrap().clear();
    model.malformed.store(3, Ordering::Relaxed);
    let report = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(report.sync.skipped.len(), 1);
    assert_eq!(report.sync.segments_embedded, 4);
    assert_eq!(report.sync.resources_updated, 0);
    let old = engine
        .search("anything")
        .paths([&path])
        .sync(false)
        .limit(20)
        .execute()
        .unwrap();
    assert!(old.hits.is_empty());
    assert_old_binding_retained();
}

#[cfg(feature = "pdf-render")]
#[test]
fn corrupt_image_is_a_page_diagnostic_and_capability_changes_retry_only_pdfs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan.pdf");
    let text = dir.path().join("note.txt");
    let cache = dir.path().join("cache.db");
    std::fs::write(&text, "orchard").unwrap();
    std::fs::write(
        &path,
        fixture::rgb_image(2, 1, &[255, 0, 0, 0, 0, 255], false),
    )
    .unwrap();
    let engine = SearchEngine::builder()
        .embedder(Model::new(false))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert_eq!(
        engine
            .index()
            .paths([&path, &text])
            .execute()
            .unwrap()
            .sync
            .page_diagnostics
            .len(),
        1
    );
    let profile = engine.profile_id().to_owned();
    drop(engine);
    let engine = SearchEngine::builder()
        .embedder(Model::new(true))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert_eq!(engine.profile_id(), profile);
    let report = engine.index().paths([&path, &text]).execute().unwrap();
    assert_eq!(report.sync.resources_updated, 1);
    assert_eq!(report.sync.resources_unchanged, 1);
    assert_eq!(report.sync.segments_embedded, 1);
    assert!(report.sync.page_diagnostics.is_empty());
    drop(engine);
    let engine = SearchEngine::builder()
        .embedder(Model::new(false))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert!(
        engine
            .search("scan")
            .paths([&path])
            .sync(false)
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
    drop(engine);
    std::fs::write(
        &path,
        fixture::broken_image("SYNTHETIC_CORRUPT_IMAGE_MARKER"),
    )
    .unwrap();
    let engine = SearchEngine::builder()
        .embedder(Model::new(true))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    let report = engine.index().paths([&path]).execute().unwrap();
    assert!(report.sync.skipped.is_empty());
    assert_eq!(report.sync.segments_embedded, 0);
    assert_eq!(
        report.sync.page_diagnostics[0].reason,
        "scanned page could not be rendered"
    );
    assert!(
        engine
            .search("scan")
            .paths([&path])
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
}

#[cfg(feature = "pdf-render")]
#[test]
fn unsupported_first_page_cannot_consume_a_filename_result_slot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    let cache = dir.path().join("cache.db");
    let text = b"BT /F1 12 Tf 50 720 Td (Public garden with basil.) Tj ET";
    std::fs::write(
        &path,
        fixture::document(&[(fixture::DRAWING, ""), (text, "")], None),
    )
    .unwrap();
    let engine = SearchEngine::builder()
        .embedder(Model::new(true))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert_eq!(
        engine
            .index()
            .paths([&path])
            .execute()
            .unwrap()
            .sync
            .segments_embedded,
        2
    );
    let profile = engine.profile_id().to_owned();
    drop(engine);
    let engine = SearchEngine::builder()
        .embedder(Model::new(false))
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert_eq!(engine.profile_id(), profile);
    for mode in [
        SearchMode::Hybrid,
        SearchMode::Lexical,
        SearchMode::Semantic,
    ] {
        let result = engine
            .search("mixed.pdf")
            .paths([&path])
            .sync(false)
            .previews(None)
            .limit(1)
            .mode(mode)
            .execute()
            .unwrap();
        assert_eq!(result.hits.len(), 1, "{mode:?}");
        assert!(
            matches!(
                result.hits[0].locator,
                ResourceLocator::PageText { page: 2, .. }
            ),
            "{mode:?}"
        );
    }
}

#[cfg(feature = "pdf-render")]
#[test]
fn image_only_cache_reuse_filters_text_pages_before_depth_and_filename_selection() {
    struct ImageOnly(Model);
    impl Embedder for ImageOnly {
        fn identity(&self) -> &ModelIdentity {
            self.0.identity()
        }
        fn supports(&self, modality: Modality) -> bool {
            modality == Modality::Image
        }
        fn embed_batch(
            &self,
            requests: &[EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<Embedding>> {
            self.0.embed_batch(requests)
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    let db = dir.path().join("cache.db");
    // More unsupported rows than the minimum candidate depth (50), followed
    // by the only supported page. A guard checking only Image fails this case.
    let text = b"BT /F1 12 Tf 50 720 Td (Public basil garden.) Tj ET";
    let mut pages = vec![(text.as_slice(), ""); 60];
    pages.push((fixture::DRAWING, ""));
    std::fs::write(&path, fixture::document(&pages, None)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(Model::new(true))
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&path]).execute().unwrap();
    let profile = engine.profile_id().to_owned();
    let resource = seyk_core::ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
    assert_eq!(
        engine
            .with_cache(|c| c.resource(&profile, &resource))
            .unwrap()
            .unwrap()
            .segments
            .len(),
        61
    );
    drop(engine);
    let engine = SearchEngine::builder()
        .embedder(ImageOnly(Model::new(true)))
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    assert_eq!(engine.profile_id(), profile);
    for mode in [SearchMode::Lexical, SearchMode::Hybrid] {
        for query in ["mixed", "mixed.pdf"] {
            let result = engine
                .search(query)
                .paths([&path])
                .sync(false)
                .previews(None)
                .limit(1)
                .mode(mode)
                .execute()
                .unwrap();
            assert_eq!(result.hits.len(), 1, "{mode:?} {query}");
            assert!(matches!(
                result.hits[0].locator,
                ResourceLocator::Page { page: 61 }
            ));
            assert!(
                result.hits[0].signals.lexical.is_some(),
                "unsupported rows must not consume lexical depth"
            );
            assert!(result.hits[0].signals.filename);
        }
    }
}
