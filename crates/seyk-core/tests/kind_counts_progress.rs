mod common;
use seyk_core::{Dimension, ProgressEvent, SearchEngine, SearchMode};
use std::sync::Mutex;

#[test]
fn returned_counts_obey_limit_and_old_results_deserialize() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..4 {
        std::fs::write(
            dir.path().join(format!("{i}.txt")),
            format!("quartz passage number {i}"),
        )
        .unwrap();
    }
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D256, "counts"))
        .build()
        .unwrap();
    for mode in [
        SearchMode::Semantic,
        SearchMode::Hybrid,
        SearchMode::Lexical,
    ] {
        for limit in [0, 1, 3, 20] {
            let plain = engine
                .search("quartz")
                .paths([dir.path()])
                .mode(mode)
                .limit(limit)
                .execute()
                .unwrap();
            let counted = engine
                .search("quartz")
                .paths([dir.path()])
                .mode(mode)
                .limit(limit)
                .kind_counts(true)
                .execute()
                .unwrap();
            assert_eq!(plain.hits, counted.hits);
            assert!(plain.kind_counts.is_none());
            let counts = counted.kind_counts.unwrap();
            assert_eq!(counts.values().sum::<usize>(), plain.hits.len());
            assert_eq!(counts.get("text").copied().unwrap_or(0), plain.hits.len());
            let json = serde_json::to_value(&plain).unwrap();
            assert!(json.get("kind_counts").is_none());
            assert!(
                serde_json::from_value::<seyk_core::SearchResults>(json)
                    .unwrap()
                    .kind_counts
                    .is_none()
            );
        }
    }
}

#[test]
fn real_indexing_reports_pending_committed_skipped_and_cached_resources() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..4 {
        std::fs::write(
            dir.path().join(format!("{i}.txt")),
            format!("quartz passage {i}"),
        )
        .unwrap();
    }
    std::fs::write(dir.path().join("unsupported.png"), b"unsupported image").unwrap();
    let engine = SearchEngine::builder()
        .batch_size(1)
        .embedder(common::FakeEmbedder::new(Dimension::D256, "progress"))
        .build()
        .unwrap();
    let events = Mutex::new(Vec::new());
    let observe = |event| {
        if let ProgressEvent::IndexProgress { progress } = event {
            events.lock().unwrap().push(progress);
        }
    };
    let report = engine
        .index()
        .paths([dir.path()])
        .progress(&observe)
        .execute()
        .unwrap();
    let snapshots = events.lock().unwrap();
    let first = snapshots.first().unwrap();
    assert_eq!(first.by_kind["text"].waiting, 4);
    assert_eq!(first.by_kind["image"].waiting, 1);
    assert_eq!(first.eta_seconds, None);
    assert!(
        snapshots
            .iter()
            .any(|s| s.by_kind["text"].indexed > 0 && s.by_kind["text"].waiting > 0)
    );
    assert!(
        snapshots
            .iter()
            .filter(|s| s.by_kind["text"].indexed > 0)
            .all(|s| s.by_kind["image"].waiting == 0)
    );
    let last = snapshots.last().unwrap();
    assert_eq!(last.by_kind["text"].indexed, 4);
    assert_eq!(last.by_kind["text"].waiting, 0);
    assert_eq!(last.by_kind["image"].skipped, 1);
    assert_eq!(last.by_kind["image"].waiting, 0);
    assert_eq!(last.eta_seconds, None);
    assert_eq!(report.sync.progress.as_ref(), Some(last));
    drop(snapshots);
    events.lock().unwrap().clear();
    let warm = engine
        .index()
        .paths([dir.path()])
        .progress(&observe)
        .execute()
        .unwrap();
    assert_eq!(warm.sync.resources_unchanged, 4);
    assert_eq!(warm.sync.progress, report.sync.progress);
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .all(|p| p.eta_seconds.is_none())
    );
    std::fs::remove_file(dir.path().join("0.txt")).unwrap();
    let after_delete = engine.index().paths([dir.path()]).execute().unwrap();
    assert_eq!(after_delete.resources_removed, 1);
    assert_eq!(
        after_delete.sync.progress.unwrap().by_kind["text"].indexed,
        3
    );
}

#[cfg(feature = "pdf")]
#[path = "support/pdf_fixture.rs"]
mod pdf_fixture;

#[cfg(feature = "pdf")]
#[test]
fn mixed_pdf_text_counts_and_cached_pdf_failures_are_distinct() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "garden basil lavender").unwrap();
    std::fs::write(
        dir.path().join("b.pdf"),
        pdf_fixture::pdf(&[pdf_fixture::PUBLIC], pdf_fixture::Kind::Plain),
    )
    .unwrap();
    std::fs::write(dir.path().join("c.pdf"), b"%PDF invalid").unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D256, "mixed"))
        .build()
        .unwrap();
    for _ in 0..2 {
        let result = engine
            .search("basil")
            .paths([dir.path()])
            .mode(SearchMode::Lexical)
            .kind_counts(true)
            .execute()
            .unwrap();
        let counts = result.kind_counts.unwrap();
        assert_eq!(counts["text"], 1);
        assert_eq!(counts["page_text"], 1);
        let progress = result.stats.sync.progress.unwrap();
        assert_eq!(progress.by_kind["pdf"].indexed, 1);
        assert_eq!(progress.by_kind["pdf"].skipped, 1);
        assert_eq!(progress.by_kind["pdf"].waiting, 0);
    }
}

#[cfg(all(unix, debug_assertions))]
#[test]
fn passes_carry_the_root_they_walked_even_if_its_alias_moves_mid_walk() {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let outside = base.join("outside");
    std::fs::create_dir(&outside).unwrap();
    for i in 0..3 {
        std::fs::write(outside.join(format!("{i}.txt")), "quartz outside").unwrap();
    }
    let alias = base.join("alias");
    std::os::unix::fs::symlink(&outside, &alias).unwrap();
    // Resolution has already chosen `outside`; swap the alias before the walk.
    let swapped = alias.clone();
    seyk_core::discovery::test_hooks::after_root_resolved(alias.clone(), move || {
        std::fs::remove_file(&swapped).unwrap();
        std::fs::create_dir(&swapped).unwrap();
    });
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D256, "walked"))
        .build()
        .unwrap();
    let events = Mutex::new(Vec::new());
    let observe = |event: ProgressEvent| events.lock().unwrap().push(event);
    engine
        .index()
        .paths([&alias])
        .progress(&observe)
        .execute()
        .unwrap();
    assert!(alias.is_dir() && !alias.is_symlink(), "hook ran");
    let events = events.into_inner().unwrap();
    assert_eq!(
        events[..2],
        [
            ProgressEvent::Walked {
                roots: vec![seyk_core::ResourceId::from_path(&outside).unwrap()],
            },
            ProgressEvent::Discovered { resources: 3 },
        ]
    );
    let Some(ProgressEvent::IndexProgress { progress }) = events
        .iter()
        .rev()
        .find(|e| matches!(e, ProgressEvent::IndexProgress { .. }))
    else {
        panic!("no counts: {events:?}");
    };
    assert_eq!(progress.by_kind["text"].indexed, 3);
}
