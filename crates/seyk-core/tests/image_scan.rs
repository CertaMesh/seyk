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

fn numeric_contract(
    root: &std::path::Path,
    images: &[Embedding],
    queries: &[Embedding],
) -> serde_json::Value {
    use seyk_embed::quant::{PhotoAnchorQ4, QueryLut};
    let dim = queries[0].dim();
    let assets = root.join("crates/seyk-embed/assets/photo-anchor");
    let parse = |path: PathBuf| {
        let bytes = std::fs::read(path).unwrap();
        let (chunks, tail) = bytes.as_chunks::<4>();
        assert!(tail.is_empty());
        chunks
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let anchor = parse(assets.join(format!("photo_anchor_{dim}.f32.bin")));
    let table = parse(assets.join(format!("quant4_{dim}.f32.bin")));
    let codec = PhotoAnchorQ4::new(dim).unwrap();
    let codes: Vec<_> = images.iter().map(|v| codec.encode(v).unwrap()).collect();
    let mut maxima = [0.0f64; 2];
    let mut failures = Vec::new();
    let tolerance = if dim == 256 { 5e-5 } else { 1.5e-4 };
    for (qi, query) in queries.iter().enumerate() {
        let lut = QueryLut::new(query).unwrap();
        for (ii, codes) in codes.iter().enumerate() {
            let actual = lut.score(codes).unwrap();
            let mut sequential = 0.0f32;
            let mut double = 0.0f64;
            for (i, &q) in query.as_slice().iter().enumerate() {
                let code = if i % 2 == 0 {
                    codes[i / 2] >> 4
                } else {
                    codes[i / 2] & 15
                };
                sequential += (anchor[i] + table[i] + f32::from(code) * table[dim + i]) * q;
                double += (f64::from(anchor[i])
                    + f64::from(table[i])
                    + f64::from(code) * f64::from(table[dim + i]))
                    * f64::from(q);
            }
            let errors = [
                f64::from((actual - sequential).abs()),
                (f64::from(actual) - double).abs(),
            ];
            for i in 0..2 {
                maxima[i] = maxima[i].max(errors[i]);
            }
            if errors.iter().any(|&e| e > tolerance) {
                failures
                    .push(serde_json::json!({"query_index":qi,"image_index":ii,"errors":errors}));
            }
        }
    }
    serde_json::json!({"dim":dim,"pairs":images.len()*queries.len(),"tolerance":tolerance,"max_errors_f32_f64":maxima,"failures":failures})
}

fn snapshot(cache_path: &std::path::Path, output: &std::path::Path) {
    let pending = output.with_extension("part");
    if pending.exists() {
        std::fs::remove_file(&pending).unwrap();
    }
    let conn = rusqlite::Connection::open(cache_path).unwrap();
    conn.execute("VACUUM INTO ?1", [pending.to_string_lossy().as_ref()])
        .unwrap();
    std::fs::rename(pending, output).unwrap();
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
    let mut metrics = Vec::new();
    let mut numerics = Vec::new();
    let mut disk = Vec::new();
    for dimension in [Dimension::D256, Dimension::D768] {
        let convert = |v: &Embedding| {
            if dimension == Dimension::D768 {
                v.clone()
            } else {
                Embedding::from_raw(v.as_slice().to_vec(), dimension).unwrap()
            }
        };
        let numeric = numeric_contract(
            &root,
            &prepared.images.iter().map(convert).collect::<Vec<_>>(),
            &prepared.text.iter().map(convert).collect::<Vec<_>>(),
        );
        if !numeric["failures"].as_array().unwrap().is_empty() {
            failures.push(numeric.clone());
        }
        numerics.push(numeric);
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
        let f32_snapshot = fixture.join(format!("p3-4517-d{}-f32.sqlite", dimension.get()));
        snapshot(&cache_path, &f32_snapshot);
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
        let q4_snapshot = fixture.join(format!("p3-4517-d{}-q4.sqlite", dimension.get()));
        snapshot(&cache_path, &q4_snapshot);
        disk.push(serde_json::json!({"dim":dimension.get(),"f32_only_bytes":std::fs::metadata(&f32_snapshot).unwrap().len(),"f32_plus_q4_bytes":std::fs::metadata(&q4_snapshot).unwrap().len(),"image_code_rows":280}));
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
            metrics.push(serde_json::json!({"dim":dimension.get(),"mode":format!("{mode:?}"),"f32_ndcg_at_10":totals[0]/60.0,"q4_ndcg_at_10":totals[1]/60.0,"drop":drop}));
            if drop > 0.005 {
                failures.push(serde_json::json!({"dim":dimension.get(),"mode":format!("{mode:?}"),"ndcg_drop":drop}));
            }
        }
        assert_eq!(embedder.documents.load(Ordering::Relaxed), 280);
    }
    std::fs::write(fixture.join("p3-correctness.json"),serde_json::to_vec_pretty(&serde_json::json!({"fingerprint":prepared.fingerprint,"numerics":numerics,"metrics":metrics,"disk":disk,"rankings":report,"failures":failures})).unwrap()).unwrap();
    assert!(
        failures.is_empty(),
        "STOP: P2 parity failed; see p3-correctness.json. Do not tune R."
    );
}

#[test]
fn q4_exact_reranking_handles_mixed_rows_ties_thresholds_queries_and_partial_coverage() {
    for dimension in [Dimension::D256, Dimension::D768] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("files");
        std::fs::create_dir(&root).unwrap();
        let mut image_vectors = HashMap::new();
        let mut vectors = Vec::new();
        for id in 0..30 {
            let input = if id == 29 {
                "image-0".to_owned()
            } else {
                format!("image-{id}")
            };
            let vector = Embedding::from_raw(
                (0..dimension.get())
                    .map(|i| {
                        if (i * 17 + id % 29 * 31) % 7 < 3 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect(),
                dimension,
            )
            .unwrap();
            image_vectors.insert(
                blake3::hash(input.as_bytes()).to_hex().to_string(),
                vector.clone(),
            );
            vectors.push(vector);
            std::fs::write(root.join(format!("image-{id:02}.jpg")), input).unwrap();
        }
        std::fs::write(root.join("notes.md"), "query").unwrap();
        let embedder = Arc::new(PreparedEmbedder {
            identity: ModelIdentity {
                model_id: "google/embeddinggemma-2".into(),
                revision: "synthetic mixed fixture".into(),
                variant: "onnx-q8;text-literals-v1".into(),
                dimension,
                normalization: "mean-pool;truncate;l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            },
            images: image_vectors,
            queries: HashMap::from([
                ("query".into(), vectors[0].clone()),
                ("also".into(), vectors[3].clone()),
            ]),
            documents: AtomicUsize::new(0),
        });
        let path = temp.path().join("cache.sqlite");
        let engine = SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .cache(SqliteCache::open(&path).unwrap())
            .build()
            .unwrap();
        engine.index().paths([&root]).execute().unwrap();
        let documents = embedder.documents.load(Ordering::Relaxed);
        engine
            .search("query")
            .paths([&root])
            .image_scan(ImageScan::PhotoAnchorQ4)
            .previews(None)
            .execute()
            .unwrap();
        assert_eq!(embedder.documents.load(Ordering::Relaxed), documents);
        let count = SqliteCache::open(&path).unwrap().stats().unwrap().profiles[0].image_codes;
        assert_eq!(
            count, 29,
            "shared input hashes retain separate segment evidence"
        );
        // Leave a mixture of missing, unknown-codec and valid current-codec rows.
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch("DELETE FROM image_codes WHERE input_hash IN (SELECT input_hash FROM image_codes ORDER BY input_hash LIMIT 8); UPDATE image_codes SET codec_id='future' WHERE input_hash IN (SELECT input_hash FROM image_codes ORDER BY input_hash LIMIT 5);").unwrap();
        drop(engine);
        let engine = SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .cache(SqliteCache::open(&path).unwrap())
            .build()
            .unwrap();
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            for limit in [0, 1, 5, 10, 20, 50] {
                for threshold in [-1.0, 0.0, 0.9, f32::NAN] {
                    let run = |scan| {
                        engine
                            .search("query")
                            .query("also")
                            .paths([&root])
                            .mode(mode)
                            .limit(limit)
                            .min_score(threshold)
                            .sync(false)
                            .previews(None)
                            .image_scan(scan)
                            .execute()
                            .unwrap()
                    };
                    assert_eq!(
                        run(ImageScan::Exact).hits,
                        run(ImageScan::PhotoAnchorQ4).hits,
                        "d{} {mode:?} limit {limit} min {threshold}",
                        dimension.get()
                    );
                }
            }
        }
        assert_eq!(
            SqliteCache::open(&path).unwrap().stats().unwrap().profiles[0].image_codes,
            21,
            "cached-only query must not backfill"
        );
    }
}

#[test]
fn explicit_lexical_degradation_does_not_attempt_a_q4_scan() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("files");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("notes.md"), "query").unwrap();
    let embedder = Arc::new(PreparedEmbedder {
        identity: ModelIdentity {
            model_id: "google/embeddinggemma-2".into(),
            revision: "synthetic fallback".into(),
            variant: "onnx-q8;text-literals-v1".into(),
            dimension: Dimension::D256,
            normalization: "mean-pool;truncate;l2".into(),
            query_prompt: String::new(),
            document_prompt: String::new(),
            max_tokens: 512,
        },
        images: HashMap::new(),
        queries: HashMap::new(),
        documents: AtomicUsize::new(0),
    });
    let engine = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .cache(SqliteCache::open(temp.path().join("cache.sqlite")).unwrap())
        .build()
        .unwrap();
    let run = |scan| {
        engine
            .search("query")
            .paths([&root])
            .sync(false)
            .lexical_fallback(true)
            .previews(None)
            .image_scan(scan)
            .execute()
            .unwrap()
    };
    let exact = run(ImageScan::Exact);
    let q4 = run(ImageScan::PhotoAnchorQ4);
    assert_eq!(q4.hits, exact.hits);
    assert_eq!(q4.hits.len(), 1);
    assert_eq!(q4.degraded, exact.degraded);
    assert!(q4.degraded.is_some());
    assert_eq!(embedder.documents.load(Ordering::Relaxed), 0);
}
