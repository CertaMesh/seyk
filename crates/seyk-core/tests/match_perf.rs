mod common;
use common::FakeEmbedder;
use seyk_core::{Dimension, MemoryCache, SearchEngine, SearchMode, SqliteCache};
use std::{fs, time::Instant};
use tempfile::TempDir;

#[test]
#[ignore = "synthetic end-to-end large-limit latency measurement"]
fn measure_match_terms() {
    let corpus = TempDir::new().unwrap();
    // 20,000 independently ranked segments; no model or private content.
    let text = (0..20000)
        .map(|i| {
            format!(
                "# Section {i}\n\nalpha bravo charlie delta echo foxtrot golf hotel item {i}\n\n"
            )
        })
        .collect::<String>();
    let path = corpus.path().join("synthetic.md");
    fs::write(&path, text).unwrap();
    for sqlite in [false, true] {
        let db = TempDir::new().unwrap();
        let builder =
            SearchEngine::builder().embedder(FakeEmbedder::new(Dimension::D256, "perf-r1"));
        let engine = if sqlite {
            builder.cache(SqliteCache::open(db.path().join("cache.db")).unwrap())
        } else {
            builder.cache(MemoryCache::new())
        }
        .build()
        .unwrap();
        engine.index().paths([&path]).execute().unwrap();
        let resource = seyk_core::ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
        assert_eq!(
            engine
                .with_cache(|c| c.resource(engine.profile_id(), &resource))
                .unwrap()
                .unwrap()
                .segments
                .len(),
            20000
        );
        for (mode, weight) in [
            (SearchMode::Lexical, 1.0),
            (SearchMode::Hybrid, 1.0),
            (SearchMode::Hybrid, 0.0),
            (SearchMode::Semantic, 1.0),
        ] {
            let mut samples = Vec::new();
            let mut digest = None;
            for round in 0..8 {
                let start = Instant::now();
                let result = engine
                    .search("alpha bravo charlie delta echo foxtrot golf hotel")
                    .paths([&path])
                    .mode(mode)
                    .lexical_weight(weight)
                    .limit(1000)
                    .sync(false)
                    .previews(None)
                    .execute()
                    .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result.hits.len(), 1000);
                let fingerprint: Vec<_> = result
                    .hits
                    .iter()
                    .map(|h| (h.score.to_bits(), &h.locator))
                    .collect();
                let hash = blake3::hash(&serde_json::to_vec(&fingerprint).unwrap())
                    .to_hex()
                    .to_string();
                assert_eq!(digest.get_or_insert(hash.clone()), &hash);
                if round > 0 {
                    samples.push(elapsed);
                }
            }
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "MATCH_PERF sqlite={sqlite} mode={mode:?} weight={weight} segments=20000 limit=1000 groups=8 median_ms={:.3} min_ms={:.3} max_ms={:.3} digest={}",
                samples[3],
                samples[0],
                samples[6],
                digest.unwrap()
            );
        }
    }
}
