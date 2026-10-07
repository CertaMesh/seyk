//! Compare a local image with text. Enable the `image` feature and provision vision weights.
use seyk_embed::{Embedder, EmbeddingTask, ResourceInput, cosine_similarity};
use seyk_embeddinggemma::EmbeddingGemma2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: image_text <image-file> [description]")?;
    let description = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "a red circle".into());
    let bytes = std::fs::read(path)?;
    let model = EmbeddingGemma2::builder().vision(true).build()?;
    let text = model.embed_text(&description, EmbeddingTask::SearchQuery)?;
    let image = model.embed(
        ResourceInput::Image { bytes: &bytes },
        EmbeddingTask::SearchDocument,
    )?;
    println!(
        "image/text similarity: {:.4}",
        cosine_similarity(&text, &image)?
    );
    Ok(())
}
