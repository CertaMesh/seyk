//! Run only in the coordinator's quiet slot. Uses the prepared P3 cache pair.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::json;
use seyk::embedder::{
    Embedder,
    gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision},
};
use seyk::{Dimension, ImageScan, ModelIdentity, SearchEngine, SearchMode, SqliteCache};
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

#[derive(Deserialize)]
struct Photo {
    id: String,
}
#[derive(Deserialize)]
struct Query {
    text: String,
}

fn main() -> Result<()> {
    ensure!(
        std::env::var("SEYK_Q4_SLOT_GO").as_deref() == Ok("4517-p3"),
        "timing requires ROOT: SLOT-GO and SEYK_Q4_SLOT_GO=4517-p3"
    );
    let dim: usize = std::env::args()
        .nth(1)
        .context("usage: photo_anchor_latency 256|768")?
        .parse()?;
    ensure!(matches!(dim, 256 | 768), "expected d256 or d768");
    let fixture = std::env::var_os("SEYK_PHOTO_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/seyk-bench/photo-fixture")
        });
    let manifest = fs::read("scripts/fixtures/photo-manifest.json")?;
    let queries = fs::read("scripts/fixtures/photo-queries.json")?;
    let photos: Vec<Photo> = serde_json::from_slice(&manifest)?;
    let queries_parsed: Vec<Query> = serde_json::from_slice(&queries)?;
    ensure!(
        photos.len() == 280 && queries_parsed.len() == 60,
        "fixture counts changed"
    );
    let prepared: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.join("prepared.json"))?)?;
    let mut fingerprint = blake3::Hasher::new();
    fingerprint.update(&manifest);
    fingerprint.update(&queries);
    let fingerprint = fingerprint.finalize().to_hex().to_string();
    ensure!(
        prepared["fingerprint"].as_str() == Some(fingerprint.as_str()),
        "prepared fixture fingerprint mismatch"
    );
    let paths: Vec<_> = photos
        .iter()
        .map(|p| fixture.join(format!("{}.jpg", p.id)))
        .collect();
    ensure!(paths.iter().all(|p| p.is_file()), "missing fixture image");
    let mut config = GemmaConfig::default();
    config.source = ModelSource::Directory(PathBuf::from(
        std::env::var_os("SEYK_MODEL").context("SEYK_MODEL must name provisioned files")?,
    ));
    config.dimension = Dimension::try_from(dim)?;
    config.precision = Precision::Q8;
    config.vision = true;
    config.audio = false;
    config.threads = Some(4);
    let embedder = Arc::new(GemmaEmbedder::load(config)?);
    let mut expected: ModelIdentity =
        serde_json::from_value(prepared["identity"]["model"].clone())?;
    expected.dimension = Dimension::try_from(dim)?;
    ensure!(
        embedder.identity() == &expected,
        "model identity changed since preparation"
    );
    let exact = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .cache(SqliteCache::open(
            fixture.join(format!("p3-4517-d{dim}-f32.sqlite")),
        )?)
        .build()?;
    let q4 = SearchEngine::builder()
        .shared_embedder(embedder)
        .cache(SqliteCache::open(
            fixture.join(format!("p3-4517-d{dim}-q4.sqlite")),
        )?)
        .build()?;
    let query = &queries_parsed[0].text;
    let mut samples = Vec::new();
    for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
        let run = |scan| {
            let engine = if scan == ImageScan::Exact {
                &exact
            } else {
                &q4
            };
            engine
                .search(query)
                .paths(&paths)
                .mode(mode)
                .image_scan(scan)
                .sync(false)
                .limit(10)
                .previews(None)
                .execute()
        };
        // Model/session initialization and warmup are outside every sample.
        let reference = run(ImageScan::Exact)?;
        ensure!(
            reference.hits == run(ImageScan::PhotoAnchorQ4)?.hits,
            "warmup parity failed"
        );
        for repetition in 0..5 {
            let order = if repetition % 2 == 0 {
                [ImageScan::Exact, ImageScan::PhotoAnchorQ4]
            } else {
                [ImageScan::PhotoAnchorQ4, ImageScan::Exact]
            };
            for scan in order {
                let start = Instant::now();
                let result = run(scan)?;
                let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
                ensure!(result.hits == reference.hits, "timed request parity failed");
                ensure!(result.stats.segments_searched == 280, "incomplete scan");
                std::hint::black_box(&result);
                samples.push(json!({"mode":format!("{mode:?}"),"scan":format!("{scan:?}"),"repetition":repetition,"milliseconds":milliseconds,"returned":result.hits.len(),"searched":result.stats.segments_searched}));
            }
        }
    }
    let report = json!({"dimension":dim,"query_index":0,"query":query,"k":10,"fixture_fingerprint":prepared["fingerprint"],"model":expected,"threads":4,"repetitions_per_mode_and_scan":5,"errors":0,"boundary":"warm cached end-to-end engine request, including real text-query inference, SQLite reads, scanning, exact rerank and fusion; excludes model load, fixture preparation, previews and warmup","samples":samples});
    fs::write(
        fixture.join(format!("p3-4517-latency-d{dim}.json")),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
