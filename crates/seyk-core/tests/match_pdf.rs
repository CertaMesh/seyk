#![cfg(feature = "pdf-render")]
mod common;
#[path = "support/scan_fixture.rs"]
mod fixture;
use common::FakeEmbedder;
use seyk_core::{
    Dimension, Embedder, Embedding, MemoryCache, Modality, ModelIdentity, ResourceLocator,
    SearchEngine, SearchMode, SqliteCache,
};
use std::fs;
use tempfile::TempDir;

struct Multimodal(FakeEmbedder);
impl Embedder for Multimodal {
    fn identity(&self) -> &ModelIdentity {
        self.0.identity()
    }
    fn supports(&self, m: Modality) -> bool {
        matches!(m, Modality::Text | Modality::Image)
    }
    fn embed_batch(
        &self,
        r: &[seyk_embed::EmbeddingRequest<'_>],
    ) -> seyk_embed::Result<Vec<Embedding>> {
        r.iter()
            .map(|_| Embedding::from_raw(vec![1.0; 256], Dimension::D256))
            .collect()
    }
}

// Originally captured on main 71f9a184; refreshed for RRF k=5.
// Covers text and page locators with both cache implementations.
#[test]
fn pdf_ranking_matches_rrf_k5_golden() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.md"), "basil garden basil garden").unwrap();
    fs::write(dir.path().join("b.md"), "lavender garden").unwrap();
    fs::write(
        dir.path().join("garden.pdf"),
        fixture::document(
            &[
                (
                    b"BT /F1 12 Tf 50 720 Td (Public garden with basil and lavender.) Tj ET",
                    "",
                ),
                (fixture::DRAWING, ""),
                (
                    b"BT /F1 12 Tf 50 720 Td (Another garden with basil.) Tj ET",
                    "",
                ),
            ],
            None,
        ),
    )
    .unwrap();
    let mut cases = Vec::new();
    let mut saw_nonfirst_keyword = false;
    let mut saw_rank_beyond_limit = false;
    let mut saw_page = false;
    let mut saw_page_text = false;
    for sqlite in [false, true] {
        let db = TempDir::new().unwrap();
        let builder = SearchEngine::builder()
            .embedder(Multimodal(FakeEmbedder::new(Dimension::D256, "pdf-golden")));
        let e = if sqlite {
            builder.cache(SqliteCache::open(db.path().join("cache.db")).unwrap())
        } else {
            builder.cache(MemoryCache::new())
        }
        .build()
        .unwrap();
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Lexical,
        ] {
            for query in ["garden.pdf", "basil", "lavender garden", "absent"] {
                let semantic = e
                    .search(query)
                    .paths([dir.path()])
                    .mode(SearchMode::Semantic)
                    .limit(10)
                    .previews(None)
                    .execute()
                    .unwrap();
                let keyword = e
                    .with_cache(|c| {
                        let paths = c.resource_index(e.profile_id())?.into_keys().collect();
                        c.lexical_search(
                            e.profile_id(),
                            &seyk_core::lexical::LexicalQuery::new([query]),
                            &paths,
                            50,
                        )?
                        .into_iter()
                        .map(|h| {
                            let locator =
                                c.resource(e.profile_id(), &h.resource)?.unwrap().segments
                                    [h.ordinal]
                                    .locator
                                    .clone();
                            Ok((h.resource, locator))
                        })
                        .collect::<seyk_core::Result<Vec<_>>>()
                    })
                    .unwrap();
                for limit in [1, 2, 10] {
                    for weight in [0.0, 1.0] {
                        let r = e
                            .search(query)
                            .paths([dir.path()])
                            .mode(mode)
                            .limit(limit)
                            .lexical_weight(weight)
                            .previews(None)
                            .execute()
                            .unwrap();
                        for h in &r.hits {
                            let expected_semantic = if mode == SearchMode::Lexical {
                                None
                            } else {
                                semantic
                                    .hits
                                    .iter()
                                    .position(|s| {
                                        s.resource == h.resource && s.locator == h.locator
                                    })
                                    .map(|i| i + 1)
                            };
                            let expected_keyword = if mode == SearchMode::Semantic
                                || (mode == SearchMode::Hybrid && weight == 0.0)
                            {
                                None
                            } else {
                                keyword
                                    .iter()
                                    .position(|(resource, locator)| {
                                        resource == &h.resource && locator == &h.locator
                                    })
                                    .map(|i| i + 1)
                            };
                            assert_eq!(h.r#match.semantic_rank, expected_semantic);
                            assert_eq!(h.r#match.keyword_rank, expected_keyword);
                            assert_eq!(h.r#match.filename_rank, h.signals.filename.then_some(1));
                            if expected_keyword.is_none() {
                                assert!(h.r#match.matched_terms.is_empty());
                            }
                            saw_nonfirst_keyword |= expected_keyword.is_some_and(|r| r > 1);
                            saw_rank_beyond_limit |= expected_semantic.is_some_and(|r| r > limit)
                                || expected_keyword.is_some_and(|r| r > limit);
                            saw_page |= matches!(h.locator, ResourceLocator::Page { .. });
                            saw_page_text |= matches!(h.locator, ResourceLocator::PageText { .. });
                        }
                        cases.push(serde_json::json!({"sqlite": sqlite, "mode":mode,"query":query,"limit":limit,"weight":weight,
                            "hits":r.hits.iter().map(|h| serde_json::json!({
                                "file":h.resource.to_file_path().unwrap().file_name().unwrap().to_str().unwrap(),
                                "locator":h.locator,"score_bits":h.score.to_bits(),"semantic_bits":h.signals.semantic.map(f32::to_bits),
                                "lexical_bits":h.signals.lexical.map(f32::to_bits),"filename":h.signals.filename,"query_index":h.query_index,
                            })).collect::<Vec<_>>() }));
                    }
                }
            }
        }
    }
    assert!(saw_page && saw_page_text);
    assert!(saw_nonfirst_keyword && saw_rank_beyond_limit);
    let expected: Vec<serde_json::Value> = include_str!("fixtures/match-pdf-rrf-k5.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(cases, expected);
}
