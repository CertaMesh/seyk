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

#[test]
fn normalize_handles_extreme_finite_scales_and_rejects_invalid_inputs() {
    for scale in [1e20_f32, 1e-30, f32::MAX, f32::from_bits(1)] {
        let mut values = [scale, scale];
        normalize(&mut values).unwrap();
        assert!((values[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-7);
        assert_eq!(values[0], values[1]);
    }
    for mut values in [vec![0.0, 0.0], vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        let before: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
        assert!(normalize(&mut values).is_err());
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before
        );
    }
}

struct Overrides(Mock);
impl Embedder for Overrides {
    fn identity(&self) -> &ModelIdentity {
        self.0.identity()
    }
    fn dimension(&self) -> Dimension {
        Dimension::D768
    }
    fn input_identity(&self, modality: Modality) -> Result<&str> {
        if modality == Modality::Audio {
            Err(Error::Embedding("identity override".into()))
        } else {
            Ok("distinct input identity")
        }
    }
    fn supports(&self, modality: Modality) -> bool {
        modality == Modality::Video
    }
    fn embed_batch(&self, _: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        Err(Error::Embedding("batch override".into()))
    }
    fn embed(&self, _: seyk_embed::ResourceInput<'_>, _: EmbeddingTask) -> Result<Embedding> {
        Err(Error::Embedding("embed override".into()))
    }
    fn embed_text(&self, _: &str, _: EmbeddingTask) -> Result<Embedding> {
        Err(Error::Embedding("text override".into()))
    }
    fn embed_query(&self, _: &str) -> Result<Embedding> {
        Err(Error::Embedding("query override".into()))
    }
    fn embed_resources(&self, _: &[seyk_embed::ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        Err(Error::Embedding("resources override".into()))
    }
    fn embed_text_document(&self, _: Option<&str>, _: &str) -> Result<Embedding> {
        Err(Error::Embedding("document override".into()))
    }
}

#[test]
fn wrappers_preserve_every_method_override() {
    fn verify(model: impl Embedder) {
        assert_eq!(model.identity().model_id, "mock");
        assert_eq!(model.dimension(), Dimension::D768);
        assert_eq!(
            model.input_identity(Modality::Text).unwrap(),
            "distinct input identity"
        );
        assert!(model.supports(Modality::Video));
        assert!(!model.supports(Modality::Text));
        let errors = [
            (
                model.input_identity(Modality::Audio).unwrap_err(),
                "identity override",
            ),
            (model.embed_batch(&[]).unwrap_err(), "batch override"),
            (
                model
                    .embed(
                        seyk_embed::ResourceInput::Text {
                            title: None,
                            text: "x",
                        },
                        EmbeddingTask::Clustering,
                    )
                    .unwrap_err(),
                "embed override",
            ),
            (
                model
                    .embed_text("x", EmbeddingTask::Clustering)
                    .unwrap_err(),
                "text override",
            ),
            (model.embed_query("x").unwrap_err(), "query override"),
            (
                model.embed_resources(&[]).unwrap_err(),
                "resources override",
            ),
            (
                model.embed_text_document(None, "x").unwrap_err(),
                "document override",
            ),
        ];
        for (error, expected) in errors {
            assert!(matches!(error, Error::Embedding(message) if message == expected));
        }
    }
    verify(Box::new(Overrides(mock(false))) as Box<dyn Embedder>);
    verify(std::sync::Arc::new(Overrides(mock(false))) as std::sync::Arc<dyn Embedder>);
}
