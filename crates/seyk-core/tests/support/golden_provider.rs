use seyk_core::discovery::ResourceKind;
use seyk_core::provider::ReadSeek;
use seyk_core::{
    Error, ProvidedResource, ResourceId, ResourceProvider, ResourceScope, ResourceVersion, Result,
};
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

// Physical cache/temp paths must not become BM25 input in exact-score goldens.
pub fn root() -> ResourceId {
    "mem://golden/".parse().unwrap()
}

pub fn name(id: &ResourceId) -> &str {
    id.as_str().rsplit('/').next().unwrap()
}

pub struct GoldenProvider(BTreeMap<ResourceId, (ResourceKind, Vec<u8>)>);

impl GoldenProvider {
    pub fn new(resources: impl IntoIterator<Item = (&'static str, ResourceKind, Vec<u8>)>) -> Self {
        Self(
            resources
                .into_iter()
                .map(|(name, kind, bytes)| {
                    let id = format!("{}{name}", root().as_str()).parse().unwrap();
                    (id, (kind, bytes))
                })
                .collect(),
        )
    }

    fn resource(&self, id: &ResourceId) -> Result<&(ResourceKind, Vec<u8>)> {
        self.0
            .get(id)
            .ok_or_else(|| Error::Config("unknown golden resource".into()))
    }
}

impl ResourceProvider for GoldenProvider {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        self.0
            .iter()
            .filter(|(id, _)| {
                scope.roots.iter().any(|root| {
                    *id == root
                        || id.as_str().strip_prefix(root.as_str()).is_some_and(|rest| {
                            root.as_str().ends_with('/') || rest.starts_with('/')
                        })
                })
            })
            .map(|(id, (kind, bytes))| {
                Ok(ProvidedResource {
                    id: id.clone(),
                    kind: *kind,
                    size_bytes: bytes.len() as u64,
                    modified_ns: None,
                    version: self.version(id)?,
                    title: Some(
                        Path::new(name(id))
                            .file_stem()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .into(),
                    ),
                })
            })
            .collect()
    }

    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        Ok(self.resource(id)?.1.clone())
    }

    fn open(&self, id: &ResourceId) -> Result<Box<dyn ReadSeek>> {
        Ok(Box::new(Cursor::new(self.read(id)?)))
    }

    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        Ok(ResourceVersion::known(
            blake3::hash(&self.resource(id)?.1).to_hex().to_string(),
        ))
    }
}
