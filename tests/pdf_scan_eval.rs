//! Fixed synthetic scan regression; local weights only, never private documents.
#![cfg(all(feature = "pdf-render", feature = "onnx"))]
#[path = "../crates/seyk-core/tests/support/scan_fixture.rs"]
mod fixture;
use seyk::{ResourceId, ResourceLocator, SearchEngine, SearchMode};
use seyk_embeddinggemma::EmbeddingGemma2;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Corpus {
    pages: Vec<Vec<String>>,
    queries: Vec<(String, usize)>,
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes to a properly allocated rusage.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful call initialized usage.
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    Some(if cfg!(target_os = "macos") {
        rss
    } else {
        rss * 1024
    })
}
#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

#[test]
#[ignore = "requires pinned local q8 vision weights; synthetic corpus only; no downloads"]
fn twenty_scanned_pages_meet_recall_gate() {
    let corpus: Corpus = serde_json::from_str(include_str!("fixtures/pdf_scans.json")).unwrap();
    assert_eq!((corpus.pages.len(), corpus.queries.len()), (20, 14));
    let dir = tempfile::tempdir().unwrap();
    let source: ResourceId = "mem://synthetic/scan".parse().unwrap();
    let mut paths = Vec::new();
    for (index, lines) in corpus.pages.iter().enumerate() {
        let lines: Vec<_> = lines.iter().map(String::as_str).collect();
        let png = seyk::pdf::render_page(&fixture::text(&lines), &source, 1).unwrap();
        let rgb = image::load_from_memory(&png).unwrap().to_rgb8();
        let path = dir.path().join(format!("page-{:02}.pdf", index + 1));
        // A new PDF has only pixels, no hidden text. Numeric filenames cannot
        // supply the semantic/title/identifier answers in this fixed corpus.
        std::fs::write(
            &path,
            fixture::rgb_image(rgb.width(), rgb.height(), rgb.as_raw(), false),
        )
        .unwrap();
        paths.push(std::fs::canonicalize(path).unwrap());
    }
    let started = Instant::now();
    let model = EmbeddingGemma2::builder()
        .dimension(256)
        .vision(true)
        .threads(1)
        .build()
        .unwrap();
    let load_ms = started.elapsed().as_millis();
    let engine = SearchEngine::builder().embedder(model).build().unwrap();
    let mut index_times = Vec::new();
    for path in &paths {
        let started = Instant::now();
        let report = engine.index().paths([path]).execute().unwrap();
        index_times.push(started.elapsed().as_millis());
        println!(
            "SCAN_INDEX page={} elapsed_ms={}",
            index_times.len(),
            index_times.last().unwrap()
        );
        assert!(report.sync.skipped.is_empty(), "{:?}", report.sync.skipped);
        assert!(report.sync.page_diagnostics.is_empty());
        assert_eq!(report.sync.segments_embedded, 1);
    }
    let mut recall = 0;
    let mut reciprocal = 0.0;
    for (query, expected) in &corpus.queries {
        let result = engine
            .search(query)
            .paths(&paths)
            .sync(false)
            .mode(SearchMode::Semantic)
            .limit(20)
            .execute()
            .unwrap();
        assert_eq!(result.hits.len(), 20);
        assert!(
            result
                .hits
                .iter()
                .all(|h| h.preview.is_none()
                    && matches!(h.locator, ResourceLocator::Page { page: 1 }))
        );
        let rank = result
            .hits
            .iter()
            .position(|h| h.resource.to_file_path().as_ref() == Some(&paths[expected - 1]))
            .map(|n| n + 1)
            .unwrap();
        recall += usize::from(rank <= 5);
        reciprocal += 1.0 / rank as f64;
        println!("SCAN_QUERY expected_page={expected} rank={rank}");
    }
    index_times.sort_unstable();
    let recall_at_five = recall as f64 / corpus.queries.len() as f64;
    println!(
        "SCAN_EVAL pages=20 queries=14 Recall@5={recall_at_five:.3} MRR@20={:.3} model_load_ms={load_ms} index_page_p50_ms={} index_page_p95_ms={} index_total_ms={} peak_process_rss_bytes={:?}",
        reciprocal / 14.0,
        index_times[9],
        index_times[18],
        index_times.iter().sum::<u128>(),
        peak_rss_bytes()
    );
    assert!(
        recall_at_five >= 0.8,
        "scan retrieval gate failed: {recall}/14"
    );
}
