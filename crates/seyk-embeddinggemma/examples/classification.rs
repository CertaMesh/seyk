//! Nearest-label routing: an embedding similarity heuristic, not generated labels.
use seyk_embed::{Embedder, EmbeddingRequest, EmbeddingTask, ResourceInput};
use seyk_embeddinggemma::EmbeddingGemma2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = [
        "billing and invoices",
        "login and account access",
        "shipping and delivery",
    ];
    let model = EmbeddingGemma2::builder().vision(false).build()?;
    let requests: Vec<_> = labels
        .iter()
        .map(|text| EmbeddingRequest {
            input: ResourceInput::Text { title: None, text },
            task: EmbeddingTask::Classification,
        })
        .collect();
    let labels_embedded = model.embed_batch(&requests)?;
    let message = "I forgot my password and cannot sign in.";
    let vector = model.embed_text(message, EmbeddingTask::Classification)?;
    let (index, score) = labels_embedded
        .iter()
        .enumerate()
        .map(|(i, label)| (i, vector.cosine(label)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("nonempty labels");
    println!("{} ({score:.4}): {message}", labels[index]);
    Ok(())
}
