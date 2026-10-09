//! Pool scheduling must preserve request association and the existing CPU identity.
mod model_support;

use seyk_embed::{Dimension, Embedder, EmbeddingRequest, EmbeddingTask, ResourceInput};
use seyk_embeddinggemma::GemmaEmbedder;

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn parallel_sessions_preserve_order_vectors_identity_and_concurrent_queries() {
    let mut config = model_support::config(Dimension::D768);
    config.threads = Some(1);
    config.batch_size = 2;
    config.max_tokens = 128;
    let serial = GemmaEmbedder::load(config.clone()).unwrap();
    config.sessions = 2;
    let parallel = GemmaEmbedder::load(config).unwrap();
    assert_eq!(serial.identity(), parallel.identity());
    let long = "A long winter beside the harbour. ".repeat(80);
    let texts = [
        long.as_str(),
        "",
        "雪の庭",
        "short",
        "short",
        "a final sentence",
        "odd tail",
    ];
    let requests: Vec<_> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| EmbeddingRequest {
            input: ResourceInput::Text {
                title: Some("Pool test"),
                text,
            },
            task: if i % 2 == 0 {
                EmbeddingTask::SearchDocument
            } else {
                EmbeddingTask::SearchQuery
            },
        })
        .collect();
    let expected = serial.embed_batch(&requests).unwrap();
    let (actual, timings) = parallel.embed_text_batch_profiled(&requests).unwrap();
    assert_eq!(actual.len(), requests.len());
    assert!(timings.tokens > 0);
    assert!(timings.padded_tokens >= timings.tokens);
    for (a, b) in actual.iter().zip(&expected) {
        for dim in Dimension::ALL {
            let a = seyk_embed::Embedding::from_raw(a.as_slice().to_vec(), dim).unwrap();
            let b = seyk_embed::Embedding::from_raw(b.as_slice().to_vec(), dim).unwrap();
            assert!(a.cosine(&b) > 0.9999, "pool drift at {dim}");
        }
    }
    assert!(parallel.embed_batch(&[]).unwrap().is_empty());
    let query = serial.embed_query("winter harbour").unwrap();
    std::thread::scope(|scope| {
        let batch = scope.spawn(|| parallel.embed_batch(&requests).unwrap());
        let queries: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| parallel.embed_query("winter harbour").unwrap()))
            .collect();
        for result in queries {
            assert!(query.cosine(&result.join().unwrap()) > 0.9999);
        }
        for (a, b) in batch.join().unwrap().iter().zip(&expected) {
            assert!(a.cosine(b) > 0.9999);
        }
    });
    let invalid = EmbeddingRequest {
        input: ResourceInput::Image { bytes: b"invalid" },
        task: EmbeddingTask::SearchDocument,
    };
    assert!(parallel.embed_batch(&[invalid]).is_err());
    assert!(query.cosine(&parallel.embed_query("winter harbour").unwrap()) > 0.9999);
}
