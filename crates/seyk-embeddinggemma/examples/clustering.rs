//! A small spherical k-means demonstration over a batch of notes.
use seyk_embed::{Embedder, Embedding, EmbeddingRequest, EmbeddingTask, ResourceInput};
use seyk_embeddinggemma::EmbeddingGemma2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let notes = [
        "Water the basil plants every morning.",
        "Tomatoes need a sunny spot in the garden.",
        "Plant mint in a pot so it does not spread.",
        "The customer cannot reset their password.",
        "A login attempt returned an expired session error.",
        "Enable two-factor authentication on your account.",
    ];
    let model = EmbeddingGemma2::builder().vision(false).build()?;
    let requests: Vec<_> = notes
        .iter()
        .map(|text| EmbeddingRequest {
            input: ResourceInput::Text { title: None, text },
            task: EmbeddingTask::Clustering,
        })
        .collect();
    let vectors = model.embed_batch(&requests)?;
    let mut centers = [vectors[0].clone(), vectors[3].clone()];
    let mut groups = vec![0; notes.len()];
    for _ in 0..10 {
        for (group, vector) in groups.iter_mut().zip(&vectors) {
            *group = usize::from(vector.cosine(&centers[1]) > vector.cosine(&centers[0]));
        }
        for (group, center) in centers.iter_mut().enumerate() {
            let mut sum = vec![0.0; model.dimension().get()];
            let mut count = 0;
            for (assigned, vector) in groups.iter().zip(&vectors) {
                if *assigned == group {
                    count += 1;
                    for (s, v) in sum.iter_mut().zip(vector.as_slice()) {
                        *s += v;
                    }
                }
            }
            if count > 0 {
                *center = Embedding::from_raw(sum, model.dimension())?;
            }
        }
    }
    for (group, text) in groups.iter().zip(notes) {
        println!("cluster {group}: {text}");
    }
    Ok(())
}
