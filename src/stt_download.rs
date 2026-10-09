//! Explicit network boundary; never called by STT inference or search.
use crate::{Error, Result};
use std::path::PathBuf;
pub fn download() -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|_| Error::SttModel {
            reason: "download client unavailable",
        })?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        seyk_stt::manifest::REPO.into(),
        hf_hub::RepoType::Model,
        seyk_stt::manifest::REVISION.into(),
    ));
    seyk_stt::manifest::install_with(&seyk_stt::manifest::default_directory()?, |asset| {
        // download() refreshes corrupt HF cache entries too; get() can return a
        // cached corrupt file forever. The staged copy is always manifest-verified.
        repo.download(&asset.path).map_err(|_| Error::SttModel {
            reason: "download failed",
        })
    })
}
