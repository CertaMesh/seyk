use seyk_embed::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Error, Modality,
    ModelIdentity, Result, cosine_similarity, dot_product, normalize,
};

#[test]
fn widths_normalization_and_numeric_serialization() {
    for dim in Dimension::ALL {
        let e = Embedding::from_raw((0..768).map(|i| (i as f32).cos()).collect(), dim).unwrap();
        assert_eq!(e.dimension(), dim.get());
        assert!(e.normalized());
        assert!(e.as_slice().iter().all(|v| v.is_finite()));
        assert!((cosine_similarity(&e, &e).unwrap() - 1.0).abs() < 1e-5);
        assert_eq!(serde_json::to_string(&dim).unwrap(), dim.get().to_string());
    }
    let zero = Embedding::from_raw(vec![0.0; 128], Dimension::D128).unwrap();
    assert!(!zero.normalized());
    assert!(cosine_similarity(&zero, &zero).is_err());
    assert!(normalize(&mut [f32::NAN]).is_err());
    assert!(Embedding::from_raw(vec![f32::INFINITY; 128], Dimension::D128).is_err());
    assert!(matches!(
        dot_product(&[1.0], &[1.0, 2.0]),
        Err(Error::DimensionMismatch { .. })
    ));
}

struct Mock {
    identity: ModelIdentity,
    wrong_count: bool,
}
impl Embedder for Mock {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    fn supports(&self, modality: Modality) -> bool {
        modality == Modality::Text
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        if self.wrong_count {
            return Ok(vec![]);
        }
        requests
            .iter()
            .map(|r| {
                let values = match r.task {
                    EmbeddingTask::SearchQuery => vec![1.0, 0.0],
                    _ => vec![0.0, 1.0],
                };
                Embedding::from_raw(values, self.dimension())
            })
            .collect()
    }
}
fn mock(wrong_count: bool) -> Mock {
    Mock {
        identity: ModelIdentity {
            model_id: "mock".into(),
            revision: "1".into(),
            variant: "test".into(),
            dimension: Dimension::try_from(2).unwrap(),
            normalization: "l2".into(),
            query_prompt: "q".into(),
            document_prompt: "d".into(),
            max_tokens: 1,
        },
        wrong_count,
    }
}
#[test]
fn object_safe_adapters_forward_tasks_and_check_counts() {
    let model: Box<dyn Embedder> = Box::new(mock(false));
    assert_eq!(model.embed_query("x").unwrap().as_slice(), &[1.0, 0.0]);
    assert_eq!(
        model.embed_text_document(None, "x").unwrap().as_slice(),
        &[0.0, 1.0]
    );
    assert!(model.embed_batch(&[]).unwrap().is_empty());
    assert!(
        mock(true)
            .embed_text("x", EmbeddingTask::Clustering)
            .is_err()
    );
}
