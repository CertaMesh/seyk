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
    download_files(revision, &config.files(), |revision, file| {
        api.repo(hf_hub::Repo::with_revision(
            repo.clone(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
        .get(file)
        .map_err(|e| Error::Embedding(format!("downloading {file}: {e}")))
    })
}

fn download_files(
    revision: &str,
    files: &[String],
    mut get: impl FnMut(&str, &str) -> Result<PathBuf>,
) -> Result<PathBuf> {
    // Resolve once through the first file, retaining hf-hub's cache-first
    // behaviour (an info() request would require a network connection).
    let first = files.first().expect("at least one file");
    let dir = snapshot_root(&get(revision, first)?, first);
    let commit = dir.file_name().and_then(|s| s.to_str()).ok_or_else(|| {
        Error::Embedding(format!("invalid model snapshot path: {}", dir.display()))
    })?;
    for file in &files[1..] {
        // Python-populated snapshots need not have refs/<sha>, which hf-hub's
        // get() requires. Reuse files in the selected snapshot directly.
        if dir.join(file).exists() {
            continue;
        }
        let path = get(commit, file)?;
        if snapshot_root(&path, file) != dir {
            return Err(Error::Embedding(format!(
                "downloading {file}: expected snapshot {commit}, got {}",
                path.display()
            )));
        }
    }
    Ok(dir)
}

fn snapshot_root(file: &Path, rel: &str) -> PathBuf {
    let depth = Path::new(rel).components().count();
    let mut p = file.to_path_buf();
    for _ in 0..depth {
        p.pop();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn files() -> Vec<String> {
        let mut config = GemmaConfig::default();
        config.vision = true;
        config.files()
    }

    fn write_file(dir: &Path, file: &str) -> PathBuf {
        let path = dir.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"synthetic model file").unwrap();
        path
    }

    #[test]
    fn moving_alias_is_only_used_for_the_first_file() {
        let cache = tempfile::tempdir().unwrap();
        let files = files();
        let mut alias = A;
        let mut requests = Vec::new();
        let dir = download_files("main", &files, |revision, file| {
            requests.push((revision.to_string(), file.to_string()));
            let commit = if revision == "main" { alias } else { revision };
            let path = write_file(&cache.path().join("snapshots").join(commit), file);
            alias = B;
            Ok(path)
        })
        .unwrap();
        assert_eq!(dir, cache.path().join("snapshots").join(A));
        assert_eq!(requests[0], ("main".into(), files[0].clone()));
        assert_eq!(requests.len(), files.len());
        for ((revision, file), expected) in requests[1..].iter().zip(&files[1..]) {
            assert_eq!(revision, A);
            assert_eq!(file, expected);
        }
        assert!(files.iter().all(|file| dir.join(file).exists()));
    }

    #[test]
    fn cached_alias_snapshot_needs_no_network_or_sha_ref() {
        let cache = tempfile::tempdir().unwrap();
        let repo = "test/synthetic";
        let files = files();
        let dir = cache
            .path()
            .join("models--test--synthetic/snapshots")
            .join(A);
        for file in &files {
            write_file(&dir, file);
        }
        hf_hub::Cache::new(cache.path().to_path_buf())
            .model(repo.into())
            .create_ref(A)
            .unwrap();
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache.path().to_path_buf())
            .with_endpoint("http://127.0.0.1:0".into())
            .with_token(None)
            .with_progress(false)
            .build()
            .unwrap();
        let downloaded = download_files("main", &files, |revision, file| {
            api.repo(hf_hub::Repo::with_revision(
                repo.into(),
                hf_hub::RepoType::Model,
                revision.into(),
            ))
            .get(file)
            .map_err(|e| Error::Embedding(e.to_string()))
        })
        .unwrap();
        assert_eq!(downloaded, dir);
        assert!(
            !cache
                .path()
                .join("models--test--synthetic/refs")
                .join(A)
                .exists()
        );
    }

    #[test]
    fn missing_pinned_file_returns_download_error() {
        let cache = tempfile::tempdir().unwrap();
        let error = download_files("main", &files(), |revision, file| {
            if revision == "main" {
                Ok(write_file(&cache.path().join("snapshots").join(A), file))
            } else {
                assert_eq!(revision, A);
                Err(Error::Embedding("missing pinned file".into()))
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("missing pinned file"));
    }

    #[test]
    fn rejects_a_file_returned_from_another_snapshot() {
        let cache = tempfile::tempdir().unwrap();
        let error = download_files("main", &files(), |revision, file| {
            let commit = if revision == "main" { A } else { B };
            Ok(write_file(
                &cache.path().join("snapshots").join(commit),
                file,
            ))
        })
        .unwrap_err();
        assert!(error.to_string().contains("expected snapshot"));
    }
}
