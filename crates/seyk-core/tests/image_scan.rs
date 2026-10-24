use seyk_core::{Dimension, EmbeddingCache, ImageScan, MemoryCache, SearchEngine, SqliteCache};

#[test]
fn q4_requires_a_persistent_cache_and_supported_model() {
    assert!("q4".parse::<ImageScan>().is_ok());
    assert!("exact".parse::<ImageScan>().is_ok());
    assert!("f32".parse::<ImageScan>().is_err());
    assert!(!MemoryCache::new().supports_image_codes());
    let temp = tempfile::tempdir().unwrap();
    let cache = SqliteCache::open(temp.path().join("cache.sqlite")).unwrap();
    assert!(cache.supports_image_codes());
    let engine = SearchEngine::lexical().unwrap();
    let error = engine
        .search("photo")
        .paths([temp.path()])
        .image_scan(ImageScan::PhotoAnchorQ4)
        .execute()
        .unwrap_err();
    assert!(error.to_string().contains("q4"));
    let _ = Dimension::D256;
}

use serde::Deserialize;
use seyk_core::{
    Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Modality, ModelIdentity, ResourceInput,
    SearchMode,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct PreparedEmbedder {
    identity: ModelIdentity,
    images: HashMap<String, Embedding>,
    queries: HashMap<String, Embedding>,
    documents: AtomicUsize,
}
impl Embedder for PreparedEmbedder {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    fn supports(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Text | Modality::Image)
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        requests
            .iter()
            .map(|r| {
                if r.task == EmbeddingTask::SearchDocument {
                    self.documents.fetch_add(1, Ordering::Relaxed);
                }
                let value = match &r.input {
                    ResourceInput::Image { bytes } => {
                        self.images.get(&blake3::hash(bytes).to_hex().to_string())
                    }
                    ResourceInput::Text { text, .. } => self.queries.get(*text),
                    _ => None,
                };
                value
                    .cloned()
                    .ok_or_else(|| seyk_embed::Error::Embedding("unprepared fixture input".into()))
            })
            .collect()
    }
}

#[derive(Deserialize)]
struct Photo {
    id: String,
}
#[derive(Deserialize)]
struct Query {
    text: String,
    relevant: BTreeMap<String, u8>,
}
#[derive(Deserialize)]
struct Prepared {
    fingerprint: String,
    identity: serde_json::Value,
    images: Vec<Embedding>,
    text: Vec<Embedding>,
}

fn fixture_path() -> PathBuf {
    std::env::var_os("SEYK_PHOTO_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/seyk-bench/photo-fixture")
        })
}
fn ndcg(hits: &seyk_core::SearchResults, query: &Query) -> f64 {
    let gain = |grade: u8| 2f64.powi(i32::from(grade)) - 1.0;
    let actual = hits
        .hits
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, h)| {
            let path = h.resource.to_file_path().unwrap();
            let id = path.file_stem().unwrap().to_str().unwrap();
            gain(*query.relevant.get(id).unwrap_or(&0)) / ((i + 2) as f64).log2()
        })
        .sum::<f64>();
    let mut ideal: Vec<_> = query.relevant.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let ideal = ideal
        .into_iter()
        .take(10)
        .enumerate()
        .map(|(i, g)| gain(g) / ((i + 2) as f64).log2())
        .sum::<f64>();
    actual / ideal
}

/// Actual engine, retained P2 vectors, fixed shortlist. No inference or performance measurements.
#[test]
#[ignore = "requires the verified, provisioned P2 fixture and prepared.json"]
fn p2_engine_contracts_at_both_dimensions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest = std::fs::read(root.join("scripts/fixtures/photo-manifest.json")).unwrap();
    let query_bytes = std::fs::read(root.join("scripts/fixtures/photo-queries.json")).unwrap();
    let photos: Vec<Photo> = serde_json::from_slice(&manifest).unwrap();
    let queries: Vec<Query> = serde_json::from_slice(&query_bytes).unwrap();
    assert_eq!((photos.len(), queries.len()), (280, 60));
    let fixture = fixture_path();
    let prepared: Prepared =
        serde_json::from_slice(&std::fs::read(fixture.join("prepared.json")).unwrap()).unwrap();
    let mut fingerprint = blake3::Hasher::new();
    fingerprint.update(&manifest);
    fingerprint.update(&query_bytes);
    assert_eq!(
        prepared.fingerprint,
        fingerprint.finalize().to_hex().to_string()
    );
    assert_eq!((prepared.images.len(), prepared.text.len()), (280, 60));
    let paths: Vec<_> = photos
        .iter()
        .map(|p| fixture.join(format!("{}.jpg", p.id)))
        .collect();
    let image_hashes: Vec<_> = paths
        .iter()
        .map(|p| {
            blake3::hash(&std::fs::read(p).unwrap())
                .to_hex()
                .to_string()
        })
        .collect();
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for dimension in [Dimension::D256, Dimension::D768] {
        let convert = |v: &Embedding| {
            if dimension == Dimension::D768 {
                v.clone()
            } else {
                Embedding::from_raw(v.as_slice().to_vec(), dimension).unwrap()
            }
        };
        let mut identity: ModelIdentity =
            serde_json::from_value(prepared.identity["model"].clone()).unwrap();
        identity.dimension = dimension;
        let embedder = Arc::new(PreparedEmbedder {
            identity,
            images: image_hashes
                .iter()
                .cloned()
                .zip(prepared.images.iter().map(convert))
                .collect(),
            queries: queries
                .iter()
                .map(|q| q.text.clone())
                .zip(prepared.text.iter().map(convert))
                .collect(),
            documents: AtomicUsize::new(0),
        });
        let temp = tempfile::tempdir().unwrap();
        let cache_path = temp.path().join("p2.sqlite");
        let engine = SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .cache(SqliteCache::open(&cache_path).unwrap())
            .build()
            .unwrap();
        engine.index().paths(&paths).execute().unwrap();
        assert_eq!(embedder.documents.load(Ordering::Relaxed), 280);
        let profile_id = engine.profile_id().to_owned();
        let cache = SqliteCache::open(&cache_path).unwrap();
        let before = cache.stats().unwrap();
        assert_eq!(before.profiles[0].image_codes, 0);
        // Cached-only reads use f32 fallback and do not write derived codes.
        let fallback = engine
            .search(&queries[0].text)
            .paths(&paths)
            .mode(SearchMode::Semantic)
            .sync(false)
            .previews(None)
            .image_scan(ImageScan::PhotoAnchorQ4)
            .execute()
            .unwrap();
        let exact = engine
            .search(&queries[0].text)
            .paths(&paths)
            .mode(SearchMode::Semantic)
            .sync(false)
            .previews(None)
            .execute()
            .unwrap();
        assert_eq!(fallback.hits, exact.hits);
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 0);
        engine
            .search(&queries[0].text)
            .paths(&paths)
            .image_scan(ImageScan::PhotoAnchorQ4)
            .previews(None)
            .execute()
            .unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 280);
        assert_eq!(
            embedder.documents.load(Ordering::Relaxed),
            280,
            "backfill must not infer"
        );
        drop(engine);
        // Every paired query now uses a reopened cache.
        let engine = SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .cache(SqliteCache::open(&cache_path).unwrap())
            .build()
            .unwrap();
        assert_eq!(engine.profile_id(), profile_id);
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            let mut totals = [0.0; 2];
            for (qindex, query) in queries.iter().enumerate() {
                for limit in [1, 5, 10, 20, 50, 0, 281] {
                    let run = |scan| {
                        engine
                            .search(&query.text)
                            .paths(&paths)
                            .mode(mode)
                            .limit(limit)
                            .sync(false)
                            .previews(None)
                            .image_scan(scan)
                            .execute()
                            .unwrap()
                    };
                    let f32 = run(ImageScan::Exact);
                    let q4 = run(ImageScan::PhotoAnchorQ4);
                    if f32.hits != q4.hits {
                        failures.push(serde_json::json!({"dim":dimension.get(),"mode":format!("{mode:?}"),"query_index":qindex,"query":query.text,"limit":limit,"f32":f32.hits,"q4":q4.hits}));
                    }
                    if limit == 10 {
                        totals[0] += ndcg(&f32, query);
                        totals[1] += ndcg(&q4, query);
                        report.push(serde_json::json!({"dim":dimension.get(),"mode":format!("{mode:?}"),"query_index":qindex,"query":query.text,"f32":f32.hits,"q4":q4.hits}));
                    }
                }
            }
            let drop = (totals[0] - totals[1]) / 60.0;
            assert!(
                drop <= 0.005,
                "d{} {mode:?} nDCG drop {drop}",
                dimension.get()
            );
        }
        assert_eq!(embedder.documents.load(Ordering::Relaxed), 280);
    }
    std::fs::write(fixture.join("p3-correctness.json"),serde_json::to_vec_pretty(&serde_json::json!({"fingerprint":prepared.fingerprint,"rankings":report,"failures":failures})).unwrap()).unwrap();
    assert!(
        failures.is_empty(),
        "STOP: P2 parity failed; see p3-correctness.json. Do not tune R."
    );
}
