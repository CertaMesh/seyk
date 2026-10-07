//! Compare two support messages without a search engine or database.
use seyk_embed::{Embedder, EmbeddingTask, cosine_similarity};
use seyk_embeddinggemma::EmbeddingGemma2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = EmbeddingGemma2::builder()
        .vision(false)
        .dimension(256)
        .build()?;
    let a = model.embed_text("customer cannot log in", EmbeddingTask::SentenceSimilarity)?;
    let b = model.embed_text(
        "user is locked out of their account",
        EmbeddingTask::SentenceSimilarity,
    )?;
    println!("cosine similarity: {:.4}", cosine_similarity(&a, &b)?);
    Ok(())
}
