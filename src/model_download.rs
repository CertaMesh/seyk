use crate::{Error, Result};
use seyk_embeddinggemma::{GemmaConfig, ModelSource};
use std::path::{Path, PathBuf};

/// Download the files `config` needs into the Hugging Face cache.
///
/// This is the only function in the crate that uses the network. It sends
/// nothing but file requests for public model weights.
pub fn download(config: &GemmaConfig) -> Result<PathBuf> {
    let ModelSource::HuggingFace { repo, revision } = &config.source else {
        return Err(Error::Config(
            "model source is a local directory; nothing to download".into(),
        ));
    };
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Embedding(format!("cannot start download: {e}")))?;
    let repo_handle = api.repo(hf_hub::Repo::with_revision(
        repo.clone(),
        hf_hub::RepoType::Model,
        revision.clone(),
    ));
    let mut dir = None;
    for f in config.files() {
        let p = repo_handle
            .get(&f)
            .map_err(|e| Error::Embedding(format!("downloading {f}: {e}")))?;
        dir.get_or_insert_with(|| snapshot_root(&p, &f));
    }
    Ok(dir.expect("at least one file"))
}

fn snapshot_root(file: &Path, rel: &str) -> PathBuf {
    let depth = Path::new(rel).components().count();
    let mut p = file.to_path_buf();
    for _ in 0..depth {
        p.pop();
    }
    p
}
