#[cfg(all(test, feature = "onnx"))]
mod split_golden_capture {
    use super::*;
    #[test]
    fn capture_pre_split_profile_and_inputs() {
        let model =
            crate::embedder::gemma::GemmaEmbedder::load(crate::embedder::gemma::GemmaConfig {
                dimension: crate::Dimension::D128,
                vision: false,
                threads: Some(1),
                ..Default::default()
            })
            .unwrap();
        let engine = SearchEngine::builder().embedder(model).build().unwrap();
        let mut hashes = serde_json::Map::new();
        for (name, title, kind, bytes) in [
            (
                "titled",
                Some("Guide"),
                ResourceKind::Text(TextFormat::Markdown),
                b"# Setup\n\nRead the [manual](https://example.com/manual) before installing.\n"
                    .as_slice(),
            ),
            (
                "untitled",
                None,
                ResourceKind::Text(TextFormat::Plain),
                b"A short note with no document title.\n".as_slice(),
            ),
            (
                "image",
                Some("diagram"),
                ResourceKind::Image,
                b"\x89PNG\r\n\x1a\nsynthetic golden bytes".as_slice(),
            ),
        ] {
            let resource = ProvidedResource {
                id: format!("test://golden/{name}").parse().unwrap(),
                kind,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                version: ResourceVersion::unknown(),
                title: title.map(String::from),
            };
            let jobs = engine.prepare(&resource, bytes.to_vec()).unwrap();
            assert_eq!(jobs.len(), 1);
            hashes.insert(name.into(), serde_json::json!(jobs[0].segment.input_hash));
        }
        let mut vector_hashes = serde_json::Map::new();
        for dimension in crate::Dimension::ALL {
            let raw = (0..768).map(|i| (i % 17) as f32 - 8.0).collect();
            let embedding = crate::Embedding::from_raw(raw, dimension).unwrap();
            let bytes: Vec<_> = embedding
                .as_slice()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            vector_hashes.insert(
                dimension.to_string(),
                serde_json::json!(blake3::hash(&bytes).to_hex().to_string()),
            );
        }
        std::fs::write("/tmp/embed-fix-pre-split-golden.json", serde_json::to_string_pretty(&serde_json::json!({
            "base": "00c1038153d86c2060ed948f57fdc8053c56d643",
            "profile": engine.profile(), "profile_id": engine.profile_id(), "input_hashes": hashes, "model_output_hashes": vector_hashes
        })).unwrap()).unwrap();
    }
}
