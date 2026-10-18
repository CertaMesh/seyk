use tokenizers::Tokenizer;

/// Split special-token spellings in text into ordinary BPE pieces. This flag
/// controls input recognition, independently of `encode(..., true)`, which must
/// still let the pinned postprocessor insert the template's BOS/EOS tokens.
/// Keep the whole prompted string together: encoding prefix/title/body chunks
/// separately would change BPE boundaries for otherwise unaffected text.
pub(super) fn plain_text_tokenizer(template: &Tokenizer) -> Tokenizer {
    let mut text = template.clone();
    text.set_encode_special_tokens(true);
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingGemma2, EmbeddingTask, GemmaConfig, ModelSource, ONNX_REVISION};
    use serde_json::{Value, json};
    use std::collections::HashSet;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/special_tokens.json")).unwrap()
    }

    // Small hermetic BPE with the pinned added vocabulary and template. The
    // ignored test below repeats the checks against the complete pinned BPE.
    fn small_tokenizer() -> Tokenizer {
        let fixture = fixture();
        let mut vocab = serde_json::Map::new();
        for token in fixture["added_tokens"].as_array().unwrap() {
            vocab.insert(
                token["content"].as_str().unwrap().into(),
                token["id"].clone(),
            );
        }
        for c in 0x20u8..=0x7e {
            vocab.insert(char::from(c).to_string(), json!(300_000 + u32::from(c)));
        }
        Tokenizer::from_bytes(
            serde_json::to_vec(&json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": fixture["added_tokens"],
                "normalizer": null, "pre_tokenizer": null,
                "post_processor": fixture["post_processor"], "decoder": null,
                "model": {"type": "BPE", "vocab": vocab, "merges": []}
            }))
            .unwrap(),
        )
        .unwrap()
    }

    fn special_ids(tokenizer: &Tokenizer) -> HashSet<u32> {
        tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id))
            .collect()
    }

    fn assert_literal_policy(template: &Tokenizer) {
        let text = plain_text_tokenizer(template);
        let specials = special_ids(template);
        assert_eq!(specials.len(), 24);
        for token in template.get_added_tokens_decoder().values() {
            assert!(token.special && !token.normalized);
            for content in [
                token.content.clone(),
                format!("before{}after", token.content),
                token.content.repeat(3),
            ] {
                for input in [
                    EmbeddingGemma2::document_input(Some(&content), &content),
                    EmbeddingGemma2::query_input(&content),
                ] {
                    let raw = text.encode(input.as_str(), false).unwrap();
                    assert!(
                        raw.get_ids().iter().all(|id| !specials.contains(id)),
                        "{input:?}"
                    );
                    let encoded = text.encode(input.as_str(), true).unwrap();
                    let ids = encoded.get_ids();
                    assert_eq!(ids[0], template.token_to_id("<bos>").unwrap());
                    assert_eq!(ids.last(), template.token_to_id("<eos>").as_ref());
                    assert_eq!(&ids[1..ids.len() - 1], raw.get_ids());
                    assert!(
                        template
                            .encode(input.as_str(), false)
                            .unwrap()
                            .get_ids()
                            .contains(&template.token_to_id(&token.content).unwrap())
                    );
                }
            }
        }
        // Trusted processor templates retain exactly their real media ids.
        for (input, tokens) in [
            (
                "<|image><|image|><|image|><image|>",
                [
                    "<bos>",
                    "<|image>",
                    "<|image|>",
                    "<|image|>",
                    "<image|>",
                    "<eos>",
                ],
            ),
            (
                "<|audio><|audio|><|audio|><audio|>",
                [
                    "<bos>",
                    "<|audio>",
                    "<|audio|>",
                    "<|audio|>",
                    "<audio|>",
                    "<eos>",
                ],
            ),
        ] {
            let expected: Vec<_> = tokens
                .iter()
                .map(|s| template.token_to_id(s).unwrap())
                .collect();
            assert_eq!(template.encode(input, true).unwrap().get_ids(), &expected);
        }
    }

    #[test]
    fn all_pinned_special_spellings_are_literal_except_template_tokens() {
        assert_literal_policy(&small_tokenizer());
    }

    #[test]
    #[ignore = "requires pinned local tokenizer only (SEYK_MODEL or local HF cache); no downloads"]
    fn pinned_tokenizer_literals_and_ordinary_corpus_ids() {
        let config = GemmaConfig {
            vision: false,
            source: std::env::var_os("SEYK_MODEL")
                .map(|p| ModelSource::Directory(p.into()))
                .unwrap_or_default(),
            ..Default::default()
        };
        let path = crate::locate(&config).unwrap().join("tokenizer.json");
        let bytes = std::fs::read(path).unwrap();
        let pinned: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(fixture()["revision"], ONNX_REVISION);
        assert_eq!(pinned["added_tokens"], fixture()["added_tokens"]);
        assert_eq!(pinned["post_processor"], fixture()["post_processor"]);
        let mut template = Tokenizer::from_bytes(bytes).unwrap();
        template.with_padding(None);
        assert_literal_policy(&template);
        let reference: Value =
            serde_json::from_str(include_str!("../tests/fixtures/reference_text.json")).unwrap();
        let mut samples = vec![
            "".to_owned(),
            " leading and trailing ".into(),
            "\n\tline\r\nnext".into(),
            "Über Größen, Ελληνικά, 日本語, العربية 🦀".into(),
            "<start_of_image> is not a pinned special token".into(),
            "x ".repeat(3000),
        ];
        for group in ["queries", "docs"] {
            samples.extend(
                reference[group]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item["text"].as_str().unwrap().to_owned()),
            );
        }
        // Parity rests on structure: encode_special_tokens only changes special
        // added-token matching; no pinned BPE merge produces a special spelling,
        // and byte_fallback is on. These samples exercise ordinary-text edges.
        const ORDINARY_TEXT: &str = "A telescope records distant galaxies.
The recipe calls for flour, water, salt, and yeast.
fn main() { println!(\"hello\"); }
SELECT name FROM documents WHERE id = 42;
https://example.test/search?q=rust&limit=10
/home/example-user/notes/meeting-2026-10-15.md
Markdown: **bold**, _italic_, `code`, and [a link](https://example.test).
JSON: {\"enabled\": true, \"count\": 3}
Numbers: 0 -17 3.14159 1e-9 100%.
Punctuation: < > | / \\ & = + ~ ^ ! ?
A title: a subtitle | a second field.
Café and café have different Unicode representations.";
        samples.extend(ORDINARY_TEXT.lines().map(str::to_owned));
        for max_length in [32, 2048] {
            template
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length,
                    ..Default::default()
                }))
                .unwrap();
            let text = plain_text_tokenizer(&template);
            for sample in &samples {
                assert!(
                    !template
                        .get_added_tokens_decoder()
                        .values()
                        .any(|t| sample.contains(&t.content))
                );
                for task in [
                    EmbeddingTask::SearchDocument,
                    EmbeddingTask::SearchQuery,
                    EmbeddingTask::Classification,
                    EmbeddingTask::Clustering,
                    EmbeddingTask::CodeRetrieval,
                    EmbeddingTask::FactChecking,
                    EmbeddingTask::QuestionAnswering,
                    EmbeddingTask::SentenceSimilarity,
                ] {
                    let input = EmbeddingGemma2::text_input(Some(sample), sample, task).unwrap();
                    let before = template.encode(input.as_str(), true).unwrap();
                    let after = text.encode(input.as_str(), true).unwrap();
                    assert_eq!(before.get_ids(), after.get_ids(), "{task:?} {sample:?}");
                    assert_eq!(before.get_attention_mask(), after.get_attention_mask());
                }
            }
        }
        eprintln!(
            "ordinary-input parity: {} corpus samples x 8 tasks x 2 truncation limits",
            samples.len()
        );
    }
}
