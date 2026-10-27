mod common;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Modality, ModelIdentity,
    SearchEngine,
};
use std::sync::{Arc, Mutex};

struct Recording {
    inner: common::FakeEmbedder,
    tasks: Mutex<Vec<(EmbeddingTask, Option<String>, String)>>,
}
impl Embedder for Recording {
    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        self.inner.supports(modality)
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        for request in requests {
            if let seyk_embed::ResourceInput::Text { title, text } = request.input {
                self.tasks.lock().unwrap().push((
                    request.task,
                    title.map(str::to_owned),
                    text.to_owned(),
                ));
            }
        }
        self.inner.embed_batch(requests)
    }
}
#[test]
fn search_uses_generic_task_requests_and_reuses_unchanged_vectors() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("garden.md"),
        "# Herbs\n\nBasil and [mint](plants.md) grow outside.\n",
    )
    .unwrap();
    let embedder = Arc::new(Recording {
        inner: common::FakeEmbedder::new(Dimension::D128, "boundary"),
        tasks: Mutex::new(vec![]),
    });
    let engine = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .build()
        .unwrap();
    let first = engine
        .search("herbs")
        .paths([dir.path()])
        .execute()
        .unwrap();
    assert!(first.stats.sync.segments_embedded > 0);
    let tasks = embedder.tasks.lock().unwrap().clone();
    assert!(
        tasks
            .iter()
            .any(|(task, _, text)| *task == EmbeddingTask::SearchQuery && text == "herbs")
    );
    assert!(
        tasks
            .iter()
            .any(|(task, title, text)| *task == EmbeddingTask::SearchDocument
                && title.as_deref() == Some("garden > Herbs")
                && !text.contains("[mint]")
                && !text.contains("plants.md"))
    );
    let second = engine
        .search("herbs")
        .paths([dir.path()])
        .execute()
        .unwrap();
    assert_eq!(second.stats.sync.segments_embedded, 0);
    assert_eq!(first.hits[0].locator, second.hits[0].locator);
}
#[test]
fn embedding_errors_keep_existing_search_variants() {
    use seyk_core::Error as C;
    use seyk_embed::Error as E;
    assert!(matches!(C::from(E::Embedding("x".into())), C::Embedding(_)));
    assert!(matches!(
        C::from(E::ModelNotFound("x".into())),
        C::ModelNotFound(_)
    ));
    assert!(matches!(C::from(E::Config("x".into())), C::Config(_)));
    assert!(matches!(
        C::from(E::DimensionMismatch {
            expected: 3,
            actual: 2
        }),
        C::DimensionMismatch {
            expected: 3,
            actual: 2
        }
    ));
    assert!(matches!(
        C::from(E::UnsupportedModality {
            modality: Modality::Audio
        }),
        C::UnsupportedModality {
            modality: Modality::Audio
        }
    ));
    assert!(matches!(
        C::from(E::Io {
            path: "test".into(),
            source: std::io::Error::other("x")
        }),
        C::Io { .. }
    ));
    assert!(matches!(
        C::from(E::UnsupportedTask {
            task: EmbeddingTask::Clustering,
            modality: Modality::Image
        }),
        C::UnsupportedTask { .. }
    ));
}

#[test]
fn dense_fixture_restart_reuses_all_cached_document_inputs() {
    let root = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let database = cache_dir.path().join("index.sqlite");
    // Invented dense numeric text exercises many character-sized segments.
    let dense = (0..8000).map(|i| format!("{i},")).collect::<String>();
    std::fs::write(root.path().join("dense.txt"), dense).unwrap();
    let embedder = Arc::new(Recording {
        inner: common::FakeEmbedder::new(Dimension::D128, "dense-restart"),
        tasks: Mutex::new(vec![]),
    });
    let build = || {
        SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .batch_size(16)
            .cache(seyk_core::SqliteCache::open(&database).unwrap())
            .build()
            .unwrap()
    };
    let first = build();
    let report = first.index().paths([root.path()]).execute().unwrap();
    assert_eq!(report.sync.resources_discovered, 1);
    assert!(report.sync.segments_embedded > 16);
    let recorded = embedder.tasks.lock().unwrap().len();
    assert!(recorded > 16);
    drop(first);
    let restarted = build();
    let report = restarted.index().paths([root.path()]).execute().unwrap();
    assert_eq!(report.sync.segments_embedded, 0);
    assert_eq!(embedder.tasks.lock().unwrap().len(), recorded);
}

#[test]
fn cached_index_diagnostic_child() {
    if std::env::var_os("SEYK_INDEX_DIAGNOSTIC_CHILD").is_none() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("quartz.txt"), "quartz orchard notes").unwrap();
    let semantic = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "diagnostics"))
        .build()
        .unwrap();
    semantic.index().paths([root.path()]).execute().unwrap();
    let cached = semantic.index().paths([root.path()]).execute().unwrap();
    assert_eq!(cached.sync.segments_embedded, 0);
    let lexical = SearchEngine::lexical().unwrap();
    lexical.index().paths([root.path()]).execute().unwrap();
    lexical.index().paths([root.path()]).execute().unwrap();
}

#[test]
fn cached_and_lexical_passes_report_zero_inference_work() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "cached_index_diagnostic_child", "--nocapture"])
        .env("SEYK_INDEX_DIAGNOSTIC_CHILD", "1")
        .env("SEYK_MEMORY_DIAGNOSTICS", "1")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let work: Vec<_> = stderr
        .lines()
        .filter(|line| line.contains("event=index_work "))
        .collect();
    assert_eq!(work.len(), 4);
    assert!(work[0].contains("missing_inputs=1 "));
    assert!(
        work[1..]
            .iter()
            .all(|line| line.contains("missing_inputs=0 "))
    );
}
