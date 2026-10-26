//! Export resumable calibration vectors and authoritative v1 scores, without downloads.
use std::{collections::BTreeMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use seyk_embed::{
    Dimension, Embedder, Embedding, EmbeddingTask, Modality, ResourceInput,
    quant::{PhotoAnchorQ4, QueryLut},
};
use seyk_embeddinggemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
struct Job {
    images: Vec<Image>,
    queries: Vec<Query>,
}
#[derive(Deserialize)]
struct Image {
    id: String,
    path: String,
    split: String,
}
#[derive(Deserialize)]
struct Query {
    text: String,
    split: String,
}
#[derive(Serialize, Deserialize)]
struct Cached {
    key: String,
    vector: Embedding,
}
#[derive(Serialize)]
struct ImageOutput {
    id: String,
    split: String,
    vectors: BTreeMap<usize, Embedding>,
    codes: BTreeMap<usize, Vec<u8>>,
}
#[derive(Serialize)]
struct QueryOutput {
    text: String,
    split: String,
    vectors: BTreeMap<usize, Embedding>,
    scores: BTreeMap<usize, Value>,
}

fn cached(
    model: &GemmaEmbedder,
    directory: &Path,
    input: ResourceInput<'_>,
    task: EmbeddingTask,
) -> Result<Embedding> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"photo-calibration-export-v1");
    hash.update(&serde_json::to_vec(model.identity())?);
    hash.update(model.input_identity(input.modality())?.as_bytes());
    hash.update(format!("{task:?}").as_bytes());
    match &input {
        ResourceInput::Image { bytes } => hash.update(bytes),
        ResourceInput::Text { text, .. } => hash.update(text.as_bytes()),
        _ => return Err("unsupported calibration input".into()),
    };
    let key = hash.finalize().to_hex().to_string();
    let path = directory.join(format!("{key}.json"));
    if path.exists() {
        let entry: Cached = serde_json::from_slice(&fs::read(path)?)?;
        if entry.key != key || entry.vector.dim() != 768 || !entry.vector.normalized() {
            return Err("invalid cached vector; remove it and resume".into());
        }
        return Ok(entry.vector);
    }
    let vector = model.embed(input, task)?;
    if vector.dim() != 768 || !vector.normalized() {
        return Err("model returned an invalid calibration vector".into());
    }
    let entry = Cached { key, vector };
    let pending = path.with_extension("part");
    fs::write(&pending, serde_json::to_vec(&entry)?)?;
    fs::rename(pending, path)?;
    Ok(entry.vector)
}

fn widths(vector: Embedding) -> Result<BTreeMap<usize, Embedding>> {
    Ok(BTreeMap::from([
        (
            256,
            Embedding::from_raw(vector.as_slice()[..256].to_vec(), Dimension::D256)?,
        ),
        (768, vector),
    ]))
}

fn synthetic(path: &Path) -> Result<()> {
    let mut results = Vec::new();
    for dim in [256, 768] {
        let dimension = Dimension::try_from(dim)?;
        let vector = |seed: usize| -> Result<Embedding> {
            let values = (0..dim)
                .map(|i| ((i * 17 + seed * 31) % 127) as f32 - 63.0)
                .collect();
            Ok(Embedding::from_raw(values, dimension)?)
        };
        let mut images = (0..64).map(vector).collect::<Result<Vec<_>>>()?;
        // Include an exact tie to lock the row-index tie-break contract.
        images.push(images[0].clone());
        let codec = PhotoAnchorQ4::new(dim)?;
        let codes = images
            .iter()
            .map(|image| codec.encode(image))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut queries = Vec::new();
        for seed in [0, 2, 79] {
            let query = vector(seed)?;
            let lut = QueryLut::new(&query)?;
            let scores = codes
                .iter()
                .map(|code| lut.score(code))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let exact: Vec<_> = images.iter().map(|image| query.cosine(image)).collect();
            queries.push(json!({"vector":query,"q4":scores,"f32":exact}));
        }
        results.push(json!({"dim":dim,"images":images,"codes":codes,"queries":queries}));
    }
    fs::write(path, serde_json::to_vec(&results)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "--synthetic" {
        return synthetic(Path::new(&args[1]));
    }
    if args.len() != 4 {
        return Err("usage: photo_calibration_export JOB.json OUTPUT.json CACHE MODEL_DIR".into());
    }
    let job_bytes = fs::read(&args[0])?;
    let job: Job = serde_json::from_slice(&job_bytes)?;
    if job.images.is_empty() || job.queries.is_empty() {
        return Err("calibration job must contain images and queries".into());
    }
    let cache = Path::new(&args[2]);
    fs::create_dir_all(cache)?;
    let mut config = GemmaConfig::default();
    config.source = ModelSource::Directory(args[3].clone().into());
    config.dimension = Dimension::D768;
    config.precision = Precision::Q8;
    config.vision = true;
    config.audio = false;
    config.threads = Some(4);
    let model = GemmaEmbedder::load(config)?;
    let identity = json!({"model":model.identity(),"image":model.input_identity(Modality::Image)?,
        "text":model.input_identity(Modality::Text)?});
    let codecs = [
        (256, PhotoAnchorQ4::new(256)?),
        (768, PhotoAnchorQ4::new(768)?),
    ];
    let mut images = Vec::new();
    for (i, image) in job.images.iter().enumerate() {
        if !matches!(image.split.as_str(), "fit" | "heldout" | "eval") {
            return Err("invalid image split".into());
        }
        let bytes = fs::read(&image.path)?;
        let vectors = widths(cached(
            &model,
            cache,
            ResourceInput::Image { bytes: &bytes },
            EmbeddingTask::SearchDocument,
        )?)?;
        let codes = codecs
            .iter()
            .map(|(dim, codec)| Ok((*dim, codec.encode(&vectors[dim])?)))
            .collect::<Result<_>>()?;
        images.push(ImageOutput {
            id: image.id.clone(),
            split: image.split.clone(),
            vectors,
            codes,
        });
        eprintln!("image {}/{}", i + 1, job.images.len());
    }
    let heldout: Vec<_> = images.iter().filter(|image| image.split != "fit").collect();
    let mut queries = Vec::new();
    for query in job.queries {
        if !matches!(query.split.as_str(), "fit" | "eval") {
            return Err("invalid query split".into());
        }
        let vectors = widths(cached(
            &model,
            cache,
            ResourceInput::Text {
                title: None,
                text: &query.text,
            },
            EmbeddingTask::SearchQuery,
        )?)?;
        let mut scores = BTreeMap::new();
        if query.split == "eval" {
            for dim in [256, 768] {
                let lut = QueryLut::new(&vectors[&dim])?;
                let q4 = heldout
                    .iter()
                    .map(|image| lut.score(&image.codes[&dim]))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let f32: Vec<_> = heldout
                    .iter()
                    .map(|image| vectors[&dim].cosine(&image.vectors[&dim]))
                    .collect();
                scores.insert(dim, json!({"q4":q4,"f32":f32}));
            }
        }
        queries.push(QueryOutput {
            text: query.text,
            split: query.split,
            vectors,
            scores,
        });
    }
    let output = json!({"job_blake3":blake3::hash(&job_bytes).to_hex().to_string(),
        "identity":identity,"images":images,"queries":queries});
    let path = Path::new(&args[1]);
    let pending = path.with_extension("part");
    fs::write(&pending, serde_json::to_vec(&output)?)?;
    fs::rename(pending, path)?;
    Ok(())
}
