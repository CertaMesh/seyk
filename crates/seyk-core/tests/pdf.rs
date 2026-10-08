#![cfg(feature = "pdf")]
mod common;
#[path = "support/pdf_fixture.rs"]
mod fixture;
use common::FakeEmbedder;
use fixture::{Kind, pdf};
use seyk_core::{Dimension, Error, ResourceLocator, SearchEngine, SearchMode, SqliteCache};
use std::sync::Arc;

#[test]
fn pages_are_exact_verified_and_cached_in_both_backends() {
    for sqlite in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.PDF");
        let texts = [
            "Garden basil and lavender.",
            "Harbor sailing and anchors.",
            "Café déjà vu.",
        ];
        std::fs::write(&path, pdf(&texts, Kind::Plain)).unwrap();
        let model = Arc::new(FakeEmbedder::new(Dimension::D128, "pdf"));
        let mut builder = SearchEngine::builder().shared_embedder(model.clone());
        if sqlite {
            builder = builder.cache(SqliteCache::open(dir.path().join("cache.db")).unwrap());
        }
        let engine = builder.build().unwrap();
        for (i, query) in ["basil", "anchors", "Café"].iter().enumerate() {
            let result = engine
                .search(*query)
                .paths([&path])
                .mode(SearchMode::Lexical)
                .execute()
                .unwrap();
            let hit = &result.hits[0];
            assert!(matches!(hit.locator,ResourceLocator::PageText {page,..} if page==i as u32+1));
            assert_eq!(hit.metadata.media_type.as_deref(), Some("application/pdf"));
            assert_eq!(engine.extract(hit).unwrap().text, texts[i]);
            assert!(hit.preview.as_ref().unwrap().contains(query));
            let mut bad = hit.clone();
            if let ResourceLocator::PageText { text_version, .. } = &mut bad.locator {
                *text_version = "unknown".into();
            }
            assert!(matches!(
                engine.extract(&bad),
                Err(Error::InvalidLocator { .. })
            ));
            bad = hit.clone();
            if let ResourceLocator::PageText { page_text_hash, .. } = &mut bad.locator {
                *page_text_hash = "bad".into();
            }
            assert!(matches!(
                engine.extract(&bad),
                Err(Error::InvalidLocator { .. })
            ));
        }
        assert_eq!(model.resource_count(), 3);
        let hit = engine
            .search("basil")
            .paths([&path])
            .previews(None)
            .execute()
            .unwrap()
            .hits
            .remove(0);
        assert!(hit.preview.is_none());
        assert_eq!(model.resource_count(), 3);
        std::fs::write(
            &path,
            pdf(
                &["Garden thyme and lavender.", texts[1], texts[2]],
                Kind::Plain,
            ),
        )
        .unwrap();
        assert!(matches!(
            engine.extract(&hit),
            Err(Error::StaleResource { .. })
        ));
    }
}

#[test]
fn errors_and_limits_do_not_stop_the_rest_of_the_corpus() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("good.txt"), "garden basil").unwrap();
    std::fs::write(
        dir.path().join("broken.pdf"),
        pdf(&["garden basil", "second"], Kind::BrokenSecond),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("invalid.pdf"),
        pdf(&[fixture::PUBLIC], Kind::InvalidUtf8),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("encrypted.pdf"),
        pdf(&[fixture::PUBLIC], Kind::Encrypted),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("many.pdf"),
        pdf(&vec!["garden"; 513], Kind::Plain),
    )
    .unwrap();
    std::fs::write(dir.path().join("blank.pdf"), pdf(&[""], Kind::Plain)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let result = engine
        .search("basil")
        .paths([dir.path()])
        .execute()
        .unwrap();
    assert_eq!(result.stats.sync.skipped.len(), 4);
    assert_eq!(result.stats.sync.page_diagnostics.len(), 1);
    assert_eq!(result.hits.len(), 1);
    assert!(
        result
            .stats
            .sync
            .skipped
            .iter()
            .any(|s| s.reason.contains("512"))
    );
    assert!(
        result
            .stats
            .sync
            .skipped
            .iter()
            .any(|s| s.reason.contains("panicked"))
    );
    assert_eq!(
        engine.extract(&result.hits[0]).unwrap().text,
        "garden basil"
    );
}

#[test]
fn mixed_pages_report_scans_and_preserve_physical_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    std::fs::write(&path, pdf(&["", "garden basil", ""], Kind::Tint)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let result = engine.search("basil").paths([path]).execute().unwrap();
    assert!(result.stats.sync.skipped.is_empty());
    assert_eq!(result.stats.sync.page_diagnostics.len(), 2);
    assert!(matches!(
        result.hits[0].locator,
        ResourceLocator::PageText { page: 2, .. }
    ));
}

#[test]
fn tounicode_multibyte_ligatures_and_invalid_ranges() {
    use pdf_extract::{Dictionary, Document, Stream};
    let mut doc = Document::load_mem(&pdf(&["ABC"], Kind::Plain)).unwrap();
    let cmap = b"begincmap 1 begincodespacerange <00> <FF> endcodespacerange 3 beginbfchar <41> <6587> <42> <D83DDE42> <43> <00660069> endbfchar endcmap";
    let map = doc.add_object(Stream::new(Dictionary::new(), cmap.to_vec()));
    doc.get_object_mut((3, 0))
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .set("ToUnicode", map);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unicode.pdf");
    std::fs::write(&path, bytes).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let hit = engine
        .search("文")
        .paths([&path])
        .execute()
        .unwrap()
        .hits
        .remove(0);
    assert_eq!(engine.extract(&hit).unwrap().text, "文🙂fi");
    for (start, end) in [(1, 3), (0, 4), (8, 3), (0, u64::MAX)] {
        let mut bad = hit.clone();
        if let ResourceLocator::PageText {
            byte_start,
            byte_end,
            ..
        } = &mut bad.locator
        {
            *byte_start = start;
            *byte_end = end;
        }
        assert!(matches!(
            engine.extract(&bad),
            Err(Error::InvalidLocator { .. })
        ));
    }
    for n in [0, 2, u32::MAX] {
        let mut bad = hit.clone();
        if let ResourceLocator::PageText { page, .. } = &mut bad.locator {
            *page = n;
        }
        assert!(matches!(engine.extract(&bad), Err(Error::PdfParse { .. })));
    }
}

#[test]
fn physical_pages_survive_duplicate_text_rotation_and_multiple_columns() {
    use pdf_extract::{Dictionary, Document, Object, Stream};
    let mut doc = Document::load_mem(&pdf(
        &["garden basil", "garden basil", "placeholder"],
        Kind::Plain,
    ))
    .unwrap();
    let page = doc.get_object_mut((7, 0)).unwrap().as_dict_mut().unwrap();
    page.set("Rotate", 90);
    page.set(
        "CropBox",
        vec![Object::Integer(0), 0.into(), 300.into(), 500.into()],
    );
    doc.objects.insert((10,0), Object::Stream(Stream::new(Dictionary::new(), b"BT /F1 12 Tf 72 720 Td (left column basil) Tj ET BT /F1 12 Tf 330 720 Td (right column harbor) Tj ET".to_vec())));
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layout.pdf");
    std::fs::write(&path, bytes).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let hits = engine
        .search("basil")
        .paths([&path])
        .mode(SearchMode::Lexical)
        .limit(10)
        .execute()
        .unwrap()
        .hits;
    assert_eq!(hits.len(), 3);
    for hit in hits {
        let ResourceLocator::PageText { page, .. } = hit.locator else {
            panic!("page locator")
        };
        let text = engine.extract(&hit).unwrap().text;
        if page < 3 {
            assert_eq!(text, "garden basil");
        } else {
            assert_eq!(text, "left column basil right column harbor");
        }
    }
}

#[test]
fn corrupt_streams_reject_the_whole_resource_and_keep_other_files() {
    for kind in [
        Kind::MissingContents,
        Kind::MissingArrayContents,
        Kind::TruncatedContents,
        Kind::BadFlate,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.pdf");
        std::fs::write(&path, pdf(&["garden basil", "harbor anchors"], Kind::Plain)).unwrap();
        let engine = SearchEngine::builder()
            .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
            .build()
            .unwrap();
        let mut hit = engine
            .search("basil")
            .paths([&path])
            .mode(SearchMode::Lexical)
            .execute()
            .unwrap()
            .hits
            .remove(0);
        let corrupt = pdf(&["garden basil", "harbor anchors"], kind);
        std::fs::write(&path, &corrupt).unwrap();
        hit.metadata.content_hash = blake3::hash(&corrupt).to_hex().to_string();
        hit.metadata.size_bytes = corrupt.len() as u64;
        // Selected page 1 is valid; corruption on page 2 still forbids extraction.
        assert!(matches!(engine.extract(&hit), Err(Error::PdfParse { .. })));
        std::fs::write(dir.path().join("good.txt"), "garden basil").unwrap();
        let fresh = SearchEngine::builder()
            .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
            .build()
            .unwrap();
        let result = fresh.search("basil").paths([dir.path()]).execute().unwrap();
        assert_eq!(result.stats.sync.resources_updated, 1);
        assert_eq!(result.stats.sync.segments_embedded, 1);
        assert_eq!(result.stats.sync.skipped.len(), 1);
        assert!(result.stats.sync.page_diagnostics.is_empty());
        assert!(
            result.stats.sync.skipped[0]
                .reason
                .starts_with("PDF parse failed:")
        );
        assert!(!result.stats.sync.skipped[0].reason.contains("corrupt.pdf"));
        assert_eq!(result.hits.len(), 1);
    }
}

#[test]
fn omitted_contents_is_a_legitimately_empty_page() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    std::fs::write(&path, pdf(&["garden basil", ""], Kind::EmptyPage)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let result = engine.search("basil").paths([&path]).execute().unwrap();
    assert_eq!(result.stats.sync.resources_updated, 1);
    assert!(result.stats.sync.skipped.is_empty());
    assert_eq!(result.stats.sync.page_diagnostics.len(), 1);
    assert_eq!(result.stats.sync.page_diagnostics[0].page, 2);
    assert_eq!(
        engine.extract(&result.hits[0]).unwrap().text,
        "garden basil"
    );
}
