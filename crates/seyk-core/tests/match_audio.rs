#![cfg(feature = "audio")]
mod common;
#[path = "support/golden_provider.rs"]
mod golden_provider;

use common::FakeEmbedder;
use golden_provider::GoldenProvider;
use seyk_core::discovery::ResourceKind;
use seyk_core::segment::TextFormat;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, MemoryCache, Modality,
    ModelIdentity, ResourceInput, ResourceLocator, SearchEngine, SearchMode, SqliteCache,
};
use tempfile::TempDir;

struct AudioEmbedder(FakeEmbedder);

impl Embedder for AudioEmbedder {
    fn identity(&self) -> &ModelIdentity {
        self.0.identity()
    }

    fn supports(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Text | Modality::Audio)
    }

    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        requests
            .iter()
            .map(|request| {
                let pair = match &request.input {
                    ResourceInput::Text { .. } if request.task == EmbeddingTask::SearchQuery => {
                        [1.0, 0.0]
                    }
                    ResourceInput::Text { .. } => [0.0, 1.0],
                    ResourceInput::Audio {
                        samples,
                        sample_rate,
                    } => {
                        assert_eq!(*sample_rate, 16000);
                        // Distinct semantic scores identify full, tail and short windows.
                        if samples.len() == 160000 {
                            [1.0, 0.0]
                        } else if samples.len() > 16000 {
                            [0.8, 0.6]
                        } else {
                            [0.6, 0.8]
                        }
                    }
                    _ => unreachable!("fixture contains only text and audio"),
                };
                let mut vector = vec![0.0; 128];
                vector[..2].copy_from_slice(&pair);
                Embedding::from_raw(vector, Dimension::D128)
            })
            .collect()
    }
}

fn wav(samples: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend((36 + samples * 2).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16000u32.to_le_bytes());
    bytes.extend(32000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend((samples * 2).to_le_bytes());
    for i in 0..samples {
        bytes.extend(((i.wrapping_mul(7919) % 32767) as i16).to_le_bytes());
    }
    bytes
}

#[test]
fn audio_ranking_and_match_reasons_match_main_golden() {
    let mut cases = Vec::new();
    let mut saw_tail = false;
    let mut saw_filename_audio = false;
    let mut saw_rank_beyond_limit = false;
    for sqlite in [false, true] {
        let db = TempDir::new().unwrap();
        let provider = GoldenProvider::new([
            ("garden.wav", ResourceKind::Audio, wav(256001)),
            ("lavender.wav", ResourceKind::Audio, wav(16000)),
            (
                "notes.md",
                ResourceKind::Text(TextFormat::Markdown),
                b"basil garden basil garden".to_vec(),
            ),
        ]);
        let builder = SearchEngine::builder()
            .provider(provider)
            .embedder(AudioEmbedder(FakeEmbedder::new(
                Dimension::D128,
                "audio-match-golden",
            )));
        let engine = if sqlite {
            builder.cache(SqliteCache::open(db.path().join("cache.db")).unwrap())
        } else {
            builder.cache(MemoryCache::new())
        }
        .build()
        .unwrap();
        engine
            .index()
            .roots([golden_provider::root()])
            .execute()
            .unwrap();
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Lexical,
        ] {
            for query in ["garden.wav", "lavender.wav", "basil", "absent"] {
                let keyword = engine
                    .with_cache(|cache| {
                        let paths = cache
                            .resource_index(engine.profile_id())?
                            .into_keys()
                            .collect();
                        cache
                            .lexical_search(
                                engine.profile_id(),
                                &seyk_core::lexical::LexicalQuery::new([query]),
                                &paths,
                                50,
                            )?
                            .into_iter()
                            .map(|hit| {
                                let locator = cache
                                    .resource(engine.profile_id(), &hit.resource)?
                                    .unwrap()
                                    .segments[hit.ordinal]
                                    .locator
                                    .clone();
                                Ok((hit.resource, locator))
                            })
                            .collect::<seyk_core::Result<Vec<_>>>()
                    })
                    .unwrap();
                for limit in [1, 2, 10] {
                    for weight in [0.0, 1.0] {
                        let result = engine
                            .search(query)
                            .roots([golden_provider::root()])
                            .mode(mode)
                            .limit(limit)
                            .lexical_weight(weight)
                            .previews(None)
                            .execute()
                            .unwrap();
                        assert!(result.stats.sync.skipped.is_empty());
                        for hit in &result.hits {
                            if let ResourceLocator::Time { start_ms, end_ms } = hit.locator {
                                assert!(hit.preview.is_none());
                                // Audio has filename/path postings but no transcript. Read
                                // ranks from the full keyword leg, before the result limit.
                                let keyword_rank = if mode == SearchMode::Semantic
                                    || (mode == SearchMode::Hybrid && weight == 0.0)
                                {
                                    None
                                } else {
                                    keyword
                                        .iter()
                                        .position(|(resource, locator)| {
                                            resource == &hit.resource && locator == &hit.locator
                                        })
                                        .map(|i| i + 1)
                                };
                                assert_eq!(hit.r#match.keyword_rank, keyword_rank);
                                assert_eq!(hit.signals.lexical.is_some(), keyword_rank.is_some());
                                assert_eq!(
                                    hit.r#match.matched_terms.is_empty(),
                                    keyword_rank.is_none()
                                );
                                let expected_rank = match (start_ms, end_ms) {
                                    (0, 10000) => 1,
                                    (8000, 16001) => 2,
                                    (0, 1000) => 3,
                                    other => panic!("unexpected synthetic audio window: {other:?}"),
                                };
                                assert_eq!(
                                    hit.r#match.semantic_rank,
                                    (mode != SearchMode::Lexical).then_some(expected_rank)
                                );
                                let named = golden_provider::name(&hit.resource) == query
                                    && start_ms == 0
                                    && mode != SearchMode::Semantic;
                                assert_eq!(hit.r#match.filename_rank, named.then_some(1));
                                assert_eq!(hit.signals.filename, named);
                                saw_tail |= start_ms == 8000;
                                saw_filename_audio |= named;
                                saw_rank_beyond_limit |=
                                    hit.r#match.semantic_rank.is_some_and(|r| r > limit);
                            }
                        }
                        cases.push(serde_json::json!({
                            "sqlite": sqlite, "mode": mode, "query": query, "limit": limit, "weight": weight,
                            "hits": result.hits.iter().map(|hit| serde_json::json!({
                                "file": golden_provider::name(&hit.resource),
                                "locator": hit.locator, "score_bits": hit.score.to_bits(),
                                "semantic_bits": hit.signals.semantic.map(f32::to_bits),
                                "lexical_bits": hit.signals.lexical.map(f32::to_bits),
                                "filename": hit.signals.filename, "query_index": hit.query_index,
                                "match": hit.r#match,
                            })).collect::<Vec<_>>()
                        }));
                    }
                }
            }
        }
    }
    assert!(saw_tail && saw_filename_audio && saw_rank_beyond_limit);
    let expected: Vec<serde_json::Value> = include_str!("fixtures/match-audio-main.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(cases, expected);
}
