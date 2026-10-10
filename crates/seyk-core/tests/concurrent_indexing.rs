// Observation hooks are intentionally absent from optimized release builds.
#![cfg(debug_assertions)]

mod common;

use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::Duration;

use common::FakeEmbedder;
use seyk_core::segment::{TextSegmenter, TextSegmenterConfig};
use seyk_core::{
    Dimension, EmbeddingCache, FsProvider, LeaseEvent, MemoryCache, ProgressEvent,
    ProvidedResource, ResourceId, ResourceProvider, ResourceScope, ResourceVersion, Result,
    SearchEngine, SearchMode, SqliteCache,
};
use tempfile::TempDir;

/// A one-shot signal: releasing a paused worker never waits for it to arrive.
/// Every receiver has a deadline, including workers being joined during unwind.
#[derive(Default)]
struct Signal {
    ready: Mutex<bool>,
    changed: Condvar,
}

impl Signal {
    fn notify(&self) {
        *self.ready.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn wait(&self, stage: &str) {
        let (ready, _) = self
            .changed
            .wait_timeout_while(
                self.ready.lock().unwrap(),
                Duration::from_secs(5),
                |ready| !*ready,
            )
            .unwrap();
        assert!(*ready, "timed out waiting for {stage}");
    }
}

#[derive(Clone, Copy)]
enum CompetingPass {
    Index,
    Search,
    Delete,
}

fn concurrent_passes(cache: impl EmbeddingCache + 'static, competing: CompetingPass) {
    let a = TempDir::new().unwrap();
    let b = TempDir::new().unwrap();
    fs::write(a.path().join("a.txt"), "alpha garden\n\nbeta workshop").unwrap();
    let b_file = b.path().join("b.txt");
    fs::write(&b_file, "quiet library").unwrap();
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "concurrent-indexing"))
        .cache(cache)
        .batch_size(1)
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 20,
            max_chars: 100,
            overlap_chars: 0,
        }))
        .build()
        .unwrap();

    if matches!(competing, CompetingPass::Delete) {
        engine.index().paths([b.path()]).execute().unwrap();
        fs::remove_file(b_file).unwrap();
    }

    // The progress event is after put_embeddings and before put_resource.
    // Pause with one published vector and no committed references for A.
    // No sleep or embedder timing is needed to reach the vulnerable interval.
    let published = Signal::default();
    let release = Signal::default();
    let progress_a = |event| {
        if matches!(event, ProgressEvent::Embedding { done: 1, total: 2 }) {
            // Callbacks must be able to inspect the cache and run cached searches:
            // neither operation may wait for the indexing lease or cache mutex.
            engine.with_cache(|cache| cache.stats()).unwrap();
            engine
                .search("library")
                .paths([b.path()])
                .sync(false)
                .mode(SearchMode::Semantic)
                .execute()
                .unwrap();
            published.notify();
            release.wait("A release");
        }
    };
    let (lease_tx, lease_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();

    let completed_while_a_pending = thread::scope(|scope| {
        let pass_a = scope.spawn(|| {
            engine
                .index()
                .paths([a.path()])
                .progress(&progress_a)
                .execute()
                .unwrap()
        });
        published.wait("A publication");
        engine.observe_indexing_lease(Arc::new(move |event| {
            lease_tx.send(event).unwrap();
        }));
        let pass_b = scope.spawn(|| {
            match competing {
                CompetingPass::Search => {
                    engine
                        .search("library")
                        .paths([b.path()])
                        .execute()
                        .unwrap();
                }
                CompetingPass::Index | CompetingPass::Delete => {
                    let report = engine.index().paths([b.path()]).execute().unwrap();
                    if matches!(competing, CompetingPass::Delete) {
                        assert_eq!(report.resources_removed, 1);
                    }
                }
            }
            done_tx.send(()).unwrap();
        });
        // Observe a failed try_lock, not merely discovery or elapsed time.
        let event = lease_rx.recv_timeout(Duration::from_secs(5));
        let completed = done_rx.try_recv().is_ok();
        release.notify();
        assert_eq!(event.unwrap(), LeaseEvent::Contended);
        let report = pass_a.join().unwrap();
        pass_b.join().unwrap();
        assert_eq!(
            lease_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            LeaseEvent::Acquired
        );
        // Later serial passes no longer need an observer.
        engine.observe_indexing_lease(Arc::new(|_| {}));
        assert_eq!(report.sync.segments_embedded, 2);
        assert_eq!(report.sync.resources_updated, 1);
        completed
    });

    // An unchanged re-index does not repair missing vectors: it skips A.
    let unchanged = engine.index().paths([a.path()]).execute().unwrap();
    assert_eq!(unchanged.sync.resources_unchanged, 1);
    assert_eq!(unchanged.sync.segments_embedded, 0);
    let found = engine
        .search("alpha garden")
        .paths([a.path()])
        .sync(false)
        .mode(SearchMode::Semantic)
        .execute()
        .unwrap();
    assert_eq!(found.stats.segments_searched, 2, "A lost a pending vector");
    assert_eq!(found.hits.len(), 2);
    assert!(found.hits.iter().any(|hit| {
        hit.preview.as_deref() == Some("alpha garden") && hit.signals.semantic.is_some()
    }));
    let stats = engine.with_cache(|cache| cache.stats()).unwrap();
    let profile = &stats.profiles[0];
    let expected = if matches!(competing, CompetingPass::Delete) {
        2
    } else {
        3
    };
    assert_eq!(profile.segments, expected);
    assert_eq!(profile.embeddings, expected);
    assert!(!completed_while_a_pending, "B bypassed the indexing lease");
}

#[test]
fn disjoint_passes_preserve_pending_vectors_in_memory() {
    for competing in [
        CompetingPass::Index,
        CompetingPass::Search,
        CompetingPass::Delete,
    ] {
        concurrent_passes(MemoryCache::new(), competing);
    }
}

#[test]
fn disjoint_passes_preserve_pending_vectors_in_sqlite() {
    for competing in [
        CompetingPass::Index,
        CompetingPass::Search,
        CompetingPass::Delete,
    ] {
        let temp = TempDir::new().unwrap();
        concurrent_passes(
            SqliteCache::open(temp.path().join("cache.db")).unwrap(),
            competing,
        );
    }
}

struct DeletionPause {
    armed: AtomicBool,
    entered: Signal,
    release: Signal,
}

impl ResourceProvider for DeletionPause {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        FsProvider.list(scope)
    }

    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        FsProvider.read(id)
    }

    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        FsProvider.version(id)
    }

    fn deletion_scope(&self, scope: &ResourceScope) -> Result<ResourceScope> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.entered.notify();
            self.release.wait("deletion release");
        }
        FsProvider.deletion_scope(scope)
    }

    fn is_deleted(&self, id: &ResourceId, scope: &ResourceScope) -> Result<bool> {
        FsProvider.is_deleted(id, scope)
    }
}

fn deletion_keeps_lease(cache: impl EmbeddingCache + 'static) {
    let a = TempDir::new().unwrap();
    let b = TempDir::new().unwrap();
    fs::write(a.path().join("a.txt"), "alpha garden").unwrap();
    let b_file = b.path().join("b.txt");
    fs::write(&b_file, "quiet library").unwrap();
    let provider = Arc::new(DeletionPause {
        armed: AtomicBool::new(false),
        entered: Signal::default(),
        release: Signal::default(),
    });
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D128, "deletion-lease"))
        .shared_provider(provider.clone())
        .cache(cache)
        .build()
        .unwrap();
    engine.index().paths([b.path()]).execute().unwrap();
    fs::remove_file(b_file).unwrap();
    provider.armed.store(true, Ordering::SeqCst);

    let (lease_tx, lease_rx) = mpsc::channel();
    let (published_tx, published_rx) = mpsc::channel();
    let release_a = Signal::default();
    let progress_a = |event| {
        if matches!(event, ProgressEvent::Embedding { done: 1, total: 1 }) {
            published_tx.send(()).unwrap();
            release_a.wait("A release");
        }
    };
    let published_during_deletion = thread::scope(|scope| {
        // B has finished sync but not deletion. A must still wait for B's lease.
        let pass_b = scope.spawn(|| engine.index().paths([b.path()]).execute().unwrap());
        provider.entered.wait("deletion entry");
        engine.observe_indexing_lease(Arc::new(move |event| {
            lease_tx.send(event).unwrap();
        }));
        let pass_a = scope.spawn(|| {
            engine
                .index()
                .paths([a.path()])
                .progress(&progress_a)
                .execute()
                .unwrap()
        });
        let event = lease_rx.recv_timeout(Duration::from_secs(5));
        let published = published_rx.try_recv().is_ok();
        provider.release.notify();
        let removed = pass_b.join().unwrap().resources_removed;
        // Without the lease A has already published; with it, A starts now.
        let publication = if published {
            Ok(())
        } else {
            published_rx.recv_timeout(Duration::from_secs(5))
        };
        release_a.notify();
        pass_a.join().unwrap();
        assert_eq!(event.unwrap(), LeaseEvent::Contended);
        assert_eq!(
            lease_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            LeaseEvent::Acquired
        );
        engine.observe_indexing_lease(Arc::new(|_| {}));
        assert!(publication.is_ok());
        assert_eq!(removed, 1);
        published
    });

    let unchanged = engine.index().paths([a.path()]).execute().unwrap();
    assert_eq!(unchanged.sync.resources_unchanged, 1);
    let found = engine
        .search("alpha garden")
        .paths([a.path()])
        .sync(false)
        .mode(SearchMode::Semantic)
        .execute()
        .unwrap();
    assert_eq!(found.hits.len(), 1, "deletion pruned A's pending vector");
    let stats = engine.with_cache(|cache| cache.stats()).unwrap();
    assert_eq!(stats.profiles[0].embeddings, 1);
    assert!(!published_during_deletion, "lease ended before deletion");
}

#[test]
fn deletion_keeps_lease_in_memory() {
    deletion_keeps_lease(MemoryCache::new());
}

#[test]
fn deletion_keeps_lease_in_sqlite() {
    let temp = TempDir::new().unwrap();
    deletion_keeps_lease(SqliteCache::open(temp.path().join("cache.db")).unwrap());
}
