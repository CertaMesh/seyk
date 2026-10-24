//! Photo Anchor quality gate. Fetch/verify with scripts/fetch_photo_fixture.py first.
//! SEYK_MODEL must point at provisioned q8 weights. No download fallback.
//! Run from the repository root; resumable vectors/results live in ~/.cache/seyk-bench/photo-fixture.
//! SEYK_PHOTO_FIXTURE overrides the fixture directory for both preparation and verification.
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command, time::Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use seyk_embed::quant::{PhotoAnchorQ4, QueryLut};
use seyk_embed::{Dimension, Embedder, Embedding, EmbeddingTask, Modality, ResourceInput};
use seyk_embeddinggemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn fixture() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("SEYK_PHOTO_FIXTURE") {
        let path = PathBuf::from(path);
        if let Ok(relative) = path.strip_prefix("~") {
            return Ok(
                PathBuf::from(std::env::var_os("HOME").ok_or("HOME is unset")?).join(relative),
            );
        }
        return Ok(path);
    }
    Ok(
        PathBuf::from(std::env::var_os("HOME").ok_or("HOME is unset")?)
            .join(".cache/seyk-bench/photo-fixture"),
    )
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
#[derive(Serialize, Deserialize)]
struct Cached {
    key: String,
    vector: Embedding,
}

fn cached(
    model: &GemmaEmbedder,
    input: ResourceInput<'_>,
    task: EmbeddingTask,
) -> Result<Embedding> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"image-quant-eval-v1");
    hash.update(&serde_json::to_vec(model.identity())?);
    hash.update(model.input_identity(input.modality())?.as_bytes());
    hash.update(format!("{task:?}").as_bytes());
    match &input {
        ResourceInput::Image { bytes } => hash.update(bytes),
        ResourceInput::Text { text, .. } => hash.update(text.as_bytes()),
        _ => return Err("unsupported fixture modality".into()),
    };
    let key = hash.finalize().to_hex().to_string();
    let path = fixture()?.join(format!("vectors/{key}.json"));
    if path.exists() {
        let entry: Cached = serde_json::from_slice(&fs::read(path)?)?;
        if entry.key != key || entry.vector.dim() != 768 || !entry.vector.normalized() {
            return Err("invalid cached vector; remove it and resume".into());
        }
        return Ok(entry.vector);
    }
    let vector = model.embed(input, task)?;
    let entry = Cached { key, vector };
    let pending = path.with_extension("part");
    fs::write(&pending, serde_json::to_vec(&entry)?)?;
    fs::rename(pending, path)?;
    Ok(entry.vector)
}

fn order(scores: &[f32]) -> Vec<usize> {
    let mut ids: Vec<_> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids
}

// Return only the exact-scored shortlist. Approximate tail scores must never
// compete against exact scores: their scales differ after quantization.
fn rerank(approx: &[f32], exact: &[f32], depth: usize) -> Vec<usize> {
    let mut ids = order(approx);
    ids.truncate(depth);
    ids.sort_unstable_by(|&a, &b| exact[b].total_cmp(&exact[a]).then(a.cmp(&b)));
    ids
}

fn metrics(ranked: &[usize], labels: &[u8]) -> (f64, f64) {
    let gain = |grade: u8| 2f64.powi(i32::from(grade)) - 1.0;
    let dcg = |grades: Vec<u8>| {
        grades
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, &grade)| gain(grade) / ((rank + 2) as f64).log2())
            .sum::<f64>()
    };
    let mut ideal = labels.to_vec();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let relevant = labels.iter().filter(|&&grade| grade > 0).count() as f64;
    (
        dcg(ranked.iter().map(|&i| labels[i]).collect()) / dcg(ideal),
        usize::from(labels[ranked[0]] > 0) as f64 / relevant,
    )
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn diagnostics(images: &[Embedding], dim: usize) -> Value {
    let (anchor, table): (&[u8], &[u8]) = match dim {
        256 => (
            include_bytes!("../crates/seyk-embed/assets/photo-anchor/photo_anchor_256.f32.bin"),
            include_bytes!("../crates/seyk-embed/assets/photo-anchor/quant4_256.f32.bin"),
        ),
        768 => (
            include_bytes!("../crates/seyk-embed/assets/photo-anchor/photo_anchor_768.f32.bin"),
            include_bytes!("../crates/seyk-embed/assets/photo-anchor/quant4_768.f32.bin"),
        ),
        _ => unreachable!(),
    };
    let (anchor, table) = (floats(anchor), floats(table));
    let mut mean = vec![0f64; dim];
    let mut clips = vec![0usize; dim];
    for image in images {
        for (i, &x) in image.as_slice().iter().enumerate() {
            mean[i] += f64::from(x) / images.len() as f64;
            let centered = x - anchor[i];
            clips[i] +=
                usize::from(centered < table[i] || centered > table[i] + 15.0 * table[dim + i]);
        }
    }
    let dot: f64 = mean
        .iter()
        .zip(&anchor)
        .map(|(&m, &a)| m * f64::from(a))
        .sum();
    let norm = |values: &[f64]| values.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine =
        dot / (norm(&mean) * norm(&anchor.iter().map(|&x| f64::from(x)).collect::<Vec<_>>()));
    let rates: Vec<_> = clips
        .iter()
        .map(|&n| n as f64 / images.len() as f64)
        .collect();
    json!({"clip_count_per_dimension":clips,"clip_denominator":images.len(),"clip_rate_per_dimension": rates,
        "clip_rate_mean": rates.iter().sum::<f64>() / dim as f64,
        "clip_rate_max": rates.iter().copied().fold(0.0, f64::max),
        "fixture_mean_anchor_cosine": cosine})
}

fn evaluate(
    images: &[Embedding],
    queries: &[Embedding],
    labels: &[Vec<u8>],
    dim: usize,
) -> Result<Value> {
    let dimension = Dimension::try_from(dim)?;
    let truncate = |vectors: &[Embedding]| -> Result<Vec<Embedding>> {
        vectors
            .iter()
            .map(|v| {
                Ok(if dim == 768 {
                    v.clone()
                } else {
                    Embedding::from_raw(v.as_slice()[..dim].to_vec(), dimension)?
                })
            })
            .collect()
    };
    let (images, queries) = (truncate(images)?, truncate(queries)?);
    let codec = PhotoAnchorQ4::new(dim)?;
    let mut codes = Vec::with_capacity(images.len() * dim / 2);
    for image in &images {
        codes.extend(codec.encode(image)?);
    }
    let mut totals = [[0f64; 4]; 4];
    let (mut error_sum, mut error_max) = (0f64, 0f64);
    let mut per_query = Vec::new();
    for (query_index, (q, relevance)) in queries.iter().zip(labels).enumerate() {
        let exact: Vec<_> = images.iter().map(|v| q.cosine(v)).collect();
        let lut = QueryLut::new(q)?;
        let approx: Vec<_> = codes
            .chunks_exact(dim / 2)
            .map(|c| lut.score(c))
            .collect::<std::result::Result<_, _>>()?;
        for (&a, &b) in approx.iter().zip(&exact) {
            let error = f64::from((a - b).abs());
            error_sum += error;
            error_max = error_max.max(error);
        }
        let ranks = [
            order(&exact),
            order(&approx),
            rerank(&approx, &exact, 20),
            rerank(&approx, &exact, 50),
        ];
        let mut row = Vec::new();
        for (i, rank) in ranks.iter().enumerate() {
            let (ndcg, recall) = metrics(rank, relevance);
            let overlap = rank[..10]
                .iter()
                .filter(|id| ranks[0][..10].contains(id))
                .count() as f64
                / 10.0;
            let identical = usize::from(rank[..10] == ranks[0][..10]) as f64;
            for (total, value) in totals[i].iter_mut().zip([ndcg, recall, overlap, identical]) {
                *total += value;
            }
            row.push(json!({"ndcg_at_10":ndcg,"recall_at_1":recall,"top10":rank[..10]}));
        }
        per_query.push(row);
    }
    for total in &mut totals {
        for value in total {
            *value /= queries.len() as f64;
        }
    }
    if totals[0][0] == 0.0 || totals[0][1] == 0.0 {
        return Err("zero f32 quality baseline".into());
    }
    let variants: Vec<_> = ["f32", "q4", "q4_rerank20", "q4_rerank50"]
        .iter()
        .enumerate()
        .map(|(i, name)| {
            json!({
        "variant":name,"ndcg_at_10":totals[i][0],"ndcg_share":totals[i][0]/totals[0][0],
        "recall_at_1":totals[i][1],"recall_share":totals[i][1]/totals[0][1],
        "top10_overlap":totals[i][2],"identical_ordered_top10_fraction":totals[i][3]})
        })
        .collect();
    let diagnostic = diagnostics(&images, dim);
    let ndcg_drop = totals[0][0] - totals[3][0];
    let quality_pass = ndcg_drop <= 0.005 && totals[3][3] == 1.0;
    Ok(
        json!({"dimension":dim,"variants":variants,"diagnostics":diagnostic,
        "quantization_error_vs_original_f32_mean":error_sum/(images.len()*queries.len()) as f64,"quantization_error_vs_original_f32_max":error_max,
        "image_vector_payload_bytes":{"f32":images.len()*dim*4,"q4":codes.len(),"rerank50_scratch_f32":50.min(images.len())*dim*4},
        "q4_buffer_capacity_bytes":codes.capacity(),
        "query_lut_payload_bytes":dim/2*256*4+4,
        "ndcg_at_10_drop_rerank50":ndcg_drop,
        "pass":{"quality":quality_pass},
        "scope":"scoring diagnostics only; actual engine contracts and Criterion scan medians are separate",
        "per_query":per_query}),
    )
}

#[derive(Serialize, Deserialize)]
struct Prepared {
    fingerprint: String,
    identity: Value,
    images: Vec<Embedding>,
    text: Vec<Embedding>,
}

fn prepare(photos: &[Photo], queries: &[Query], fingerprint: String) -> Result<Prepared> {
    fs::create_dir_all(fixture()?.join("vectors"))?;
    let mut config = GemmaConfig::default();
    config.source = ModelSource::Directory(
        std::env::var("SEYK_MODEL")
            .map_err(|_| "SEYK_MODEL must name provisioned q8 weights")?
            .into(),
    );
    config.dimension = Dimension::D768;
    config.precision = Precision::Q8;
    config.vision = true;
    config.audio = false;
    config.threads = Some(4);
    let model = GemmaEmbedder::load(config)?;
    let identity = json!({"model":model.identity(),"image":model.input_identity(Modality::Image)?});
    let started = Instant::now();
    let mut images = Vec::new();
    for (i, photo) in photos.iter().enumerate() {
        if started.elapsed().as_secs() >= 2100 {
            return Err("35-minute embedding budget reached; cached progress is resumable".into());
        }
        let bytes = fs::read(fixture()?.join(format!("{}.jpg", photo.id)))?;
        images.push(cached(
            &model,
            ResourceInput::Image { bytes: &bytes },
            EmbeddingTask::SearchDocument,
        )?);
        eprintln!("image {}/{}: {}", i + 1, photos.len(), photo.id);
    }
    let text = queries
        .iter()
        .map(|q| {
            cached(
                &model,
                ResourceInput::Text {
                    title: None,
                    text: &q.text,
                },
                EmbeddingTask::SearchQuery,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    drop(model);
    Ok(Prepared {
        fingerprint,
        identity,
        images,
        text,
    })
}

fn main() -> Result<()> {
    if !Command::new("python3")
        .args(["scripts/fetch_photo_fixture.py", "--verify"])
        .status()?
        .success()
    {
        return Err("fixture verification failed".into());
    }
    let photos: Vec<Photo> =
        serde_json::from_slice(&fs::read("scripts/fixtures/photo-manifest.json")?)?;
    let queries: Vec<Query> =
        serde_json::from_slice(&fs::read("scripts/fixtures/photo-queries.json")?)?;
    if photos.len() < 50 || queries.is_empty() {
        return Err("empty/undersized fixture".into());
    }
    let labels: Vec<Vec<u8>> = queries
        .iter()
        .map(|q| {
            if q.relevant.is_empty()
                || q.relevant.iter().any(|(id, &grade)| {
                    grade == 0 || grade > 3 || !photos.iter().any(|p| &p.id == id)
                })
            {
                return Err("invalid relevance labels".into());
            }
            Ok(photos
                .iter()
                .map(|p| q.relevant.get(&p.id).copied().unwrap_or(0))
                .collect())
        })
        .collect::<Result<_>>()?;
    let mut fingerprint = blake3::Hasher::new();
    for path in [
        "scripts/fixtures/photo-manifest.json",
        "scripts/fixtures/photo-queries.json",
    ] {
        fingerprint.update(&fs::read(path)?);
    }
    let fingerprint = fingerprint.finalize().to_hex().to_string();
    let cache = fixture()?.join("prepared.json");
    let prepared: Prepared = match std::env::args().nth(1).as_deref() {
        Some("--score-only") => serde_json::from_slice(&fs::read(&cache)?)?,
        None | Some("--prepare-only") => {
            let prepared = prepare(&photos, &queries, fingerprint.clone())?;
            let pending = cache.with_extension("part");
            fs::write(&pending, serde_json::to_vec(&prepared)?)?;
            fs::rename(pending, &cache)?;
            prepared
        }
        _ => return Err("usage: image_quant_eval [--score-only|--prepare-only]".into()),
    };
    if prepared.fingerprint != fingerprint
        || prepared.images.len() != photos.len()
        || prepared.text.len() != queries.len()
        || prepared
            .images
            .iter()
            .chain(&prepared.text)
            .any(|v| v.dim() != 768 || !v.normalized())
    {
        return Err("prepared vectors do not match this fixture".into());
    }
    if std::env::args().nth(1).as_deref() == Some("--prepare-only") {
        println!(
            "Prepared {} photos and {} queries without scoring",
            photos.len(),
            queries.len()
        );
        return Ok(());
    }
    let results = json!({"fixture_and_queries_blake3":fingerprint,
        "identity":prepared.identity,"photos":photos.len(),"queries":queries.len(),
        "dimensions":[evaluate(&prepared.images,&prepared.text,&labels,256)?,evaluate(&prepared.images,&prepared.text,&labels,768)?]});
    fs::write(
        fixture()?.join("results.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    println!("{results}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graded_metrics_and_recall_denominator() {
        let (ndcg, recall) = metrics(&[0, 1, 2], &[3, 1, 0]);
        assert_eq!(ndcg, 1.0);
        assert_eq!(recall, 0.5);
        assert_eq!(metrics(&[2, 0, 1], &[3, 1, 0]).1, 0.0);
    }
    #[test]
    fn rerank_excludes_approximate_tail_and_resolves_ties() {
        assert_eq!(rerank(&[0.9, 0.8, 0.7], &[0.1, 0.2, 0.99], 2), vec![1, 0]);
        assert_eq!(order(&[0.1, 0.1, 0.2]), vec![2, 0, 1]);
    }
}
