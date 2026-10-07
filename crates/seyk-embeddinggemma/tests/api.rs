use seyk_embeddinggemma::{
    Dimension, Embedder, EmbeddingGemma2, EmbeddingRequest, EmbeddingTask, Error, GemmaConfig,
    ModelSource, ResourceInput,
};

#[test]
fn exact_official_task_inputs_and_titles() {
    use EmbeddingTask::*;
    for (task, prefix) in [
        (SearchQuery, "task: search result | query: "),
        (SearchDocument, "title: none | text: "),
        (Classification, "task: classification | query: "),
        (Clustering, "task: clustering | query: "),
        (CodeRetrieval, "task: code retrieval | query: "),
        (FactChecking, "task: fact checking | query: "),
        (QuestionAnswering, "task: question answering | query: "),
        (SentenceSimilarity, "task: sentence similarity | query: "),
    ] {
        assert_eq!(
            EmbeddingGemma2::text_input(None, "example", task).unwrap(),
            format!("{prefix}example")
        );
    }
    assert_eq!(
        EmbeddingGemma2::text_input(Some("A\r\nB"), "x", SearchDocument).unwrap(),
        "title: A  B | text: x"
    );
    assert_eq!(
        EmbeddingGemma2::text_input(Some(" \n"), "x", SearchDocument).unwrap(),
        "title: none | text: x"
    );
    assert_eq!(
        EmbeddingGemma2::text_input(Some("ignored"), "x", SearchQuery).unwrap(),
        EmbeddingGemma2::query_input("x")
    );
}

#[test]
fn invalid_config_and_missing_local_files_fail_without_download() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        EmbeddingGemma2::builder()
            .model_path(dir.path())
            .dimension(300)
            .build(),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        EmbeddingGemma2::builder()
            .model_path(dir.path())
            .dimension(0)
            .build(),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        EmbeddingGemma2::builder()
            .model_path(dir.path())
            .batch_size(0)
            .build(),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        EmbeddingGemma2::builder()
            .model_path(dir.path())
            .vision(false)
            .build(),
        Err(Error::ModelNotFound(_))
    ));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
#[ignore = "needs local EmbeddingGemma 2 weights"]
fn batch_tasks_order_stability_and_unsupported_inputs() {
    let mut config = GemmaConfig {
        vision: false,
        dimension: Dimension::D128,
        batch_size: 2,
        ..Default::default()
    };
    if let Some(path) = std::env::var_os("SEYK_MODEL") {
        config.source = ModelSource::Directory(path.into());
    }
    let model = EmbeddingGemma2::load(config).unwrap();
    let tasks = [
        EmbeddingTask::SearchQuery,
        EmbeddingTask::Clustering,
        EmbeddingTask::Classification,
        EmbeddingTask::SentenceSimilarity,
    ];
    let texts = [
        "cat",
        "a much longer text about gardening and growing vegetables",
        "customer login problem",
        "cat",
    ];
    let requests: Vec<_> = texts
        .iter()
        .zip(tasks)
        .map(|(text, task)| EmbeddingRequest {
            input: ResourceInput::Text { title: None, text },
            task,
        })
        .collect();
    let outputs = model.embed_batch(&requests).unwrap();
    assert_eq!(outputs.len(), requests.len());
    for ((out, text), task) in outputs.iter().zip(texts).zip(tasks) {
        assert_eq!(out.dim(), 128);
        assert!(out.normalized());
        assert!(out.as_slice().iter().all(|v| v.is_finite()));
        let single = model.embed_text(text, task).unwrap();
        assert!(out.cosine(&single) > 0.999);
    }
    assert!(model.embed_batch(&[]).unwrap().is_empty());
    for input in [
        ResourceInput::Audio {
            samples: &[],
            sample_rate: 16000,
        },
        ResourceInput::Video { frames: &[] },
        ResourceInput::Image { bytes: &[] },
    ] {
        assert!(matches!(
            model.embed(input, EmbeddingTask::SearchDocument),
            Err(Error::UnsupportedModality { .. })
        ));
    }
}
