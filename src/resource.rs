use std::path::{Path, PathBuf};

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// The kind of content a resource holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
        })
    }
}

/// A canonical absolute URI identifying a resource, serialized as a string.
/// File URIs preserve non-UTF-8 Unix path bytes through percent encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResourceId(String);

impl ResourceId {
    /// Convert an absolute local path without accessing the filesystem.
    /// URI dot segments are normalized. Request `.paths(...)` resolves actual
    /// filesystem paths (including symlinks) before converting them to IDs.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let url = url::Url::from_file_path(path.as_ref())
            .map_err(|_| Error::Config("resource path must be absolute".into()))?;
        if url
            .host_str()
            .is_some_and(|host| !host.is_empty() && host != "localhost")
        {
            return Err(Error::Config(
                "resource path must be local (UNC hosts are unsupported)".into(),
            ));
        }
        if url.path().to_ascii_lowercase().contains("%00") {
            return Err(Error::Config("resource path contains NUL".into()));
        }
        let canonical = url::Url::parse(url.as_str()).map_err(|e| Error::Config(e.to_string()))?;
        Ok(Self(canonical.into()))
    }

    pub(crate) fn labels(&self) -> (String, String) {
        if let Some(path) = self.to_file_path() {
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            (name, path.to_string_lossy().into_owned())
        } else {
            let url = url::Url::parse(&self.0).expect("validated URI");
            let name = url.path().rsplit('/').next().unwrap_or("").to_string();
            (name, self.0.clone())
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn scheme(&self) -> &str {
        self.0.split_once(':').expect("validated URI").0
    }

    /// Return the local path for a file URI; other schemes have no local path.
    pub fn to_file_path(&self) -> Option<PathBuf> {
        if self.scheme() != "file" {
            return None;
        }
        url::Url::parse(&self.0).ok()?.to_file_path().ok()
    }
}

impl std::str::FromStr for ResourceId {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(Error::Config(
                "resource URI contains whitespace or controls".into(),
            ));
        }
        let bytes = value.as_bytes();
        for (i, b) in bytes.iter().enumerate() {
            if *b == b'%'
                && (i + 2 >= bytes.len()
                    || !bytes[i + 1].is_ascii_hexdigit()
                    || !bytes[i + 2].is_ascii_hexdigit())
            {
                return Err(Error::Config("invalid URI percent escape".into()));
            }
        }
        let url = url::Url::parse(value)
            .map_err(|e| Error::Config(format!("invalid resource URI: {e}")))?;
        if url.scheme() == "file" {
            if url.query().is_some()
                || url.fragment().is_some()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.port().is_some()
                || url
                    .host_str()
                    .is_some_and(|h| h != "localhost" && !h.is_empty())
            {
                return Err(Error::Config(
                    "file URI must identify a local path without query or fragment".into(),
                ));
            }
            let path = url
                .to_file_path()
                .map_err(|_| Error::Config("invalid local file URI".into()))?;
            return Self::from_path(path);
        }
        Ok(Self(url.into()))
    }
}

impl TryFrom<String> for ResourceId {
    type Error = Error;
    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}
impl From<ResourceId> for String {
    fn from(value: ResourceId) -> Self {
        value.0
    }
}
impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl Ord for ResourceId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.to_file_path(), other.to_file_path()) {
            (Some(a), Some(b)) => a.cmp(&b).then_with(|| self.0.cmp(&other.0)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => self.0.cmp(&other.0),
        }
    }
}
impl PartialOrd for ResourceId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Where inside a resource a hit was found. Deliberately modality-independent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceLocator {
    /// A byte range of a text resource. Lines are 1-based and inclusive.
    Text {
        byte_start: u64,
        byte_end: u64,
        line_start: Option<u64>,
        line_end: Option<u64>,
    },
    /// A page of a paged document (reserved for PDF and similar).
    Page { page: u32 },
    /// A time window of an audio or video resource.
    Time { start_ms: u64, end_ms: u64 },
    /// Rectangle in original-image pixels. Origin is top-left; x grows right,
    /// y grows down. Reserved: no current provider produces this locator.
    ImageRegion {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// The resource as a whole, for example a single image.
    WholeResource,
}

/// Facts about a resource at the time it was embedded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceMetadata {
    pub modality: Modality,
    pub size_bytes: u64,
    /// Modification time in nanoseconds since the Unix epoch, when available.
    pub modified_ns: Option<i64>,
    /// BLAKE3 hash of the full resource content, hex encoded.
    pub content_hash: String,
}

// JSON's path is a display aid; ResourceId is the lossless identity.
pub(crate) fn serialize_optional_path<S: serde::Serializer>(
    path: &Option<PathBuf>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    path.as_ref()
        .map(|p| p.to_string_lossy())
        .serialize(serializer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_and_serde_round_trip() {
        for path in ["/tmp/plain.md", "/tmp/hello world/文書 #?%.md"] {
            let id = ResourceId::from_path(path).unwrap();
            assert_eq!(id.to_file_path().unwrap(), Path::new(path));
            assert_eq!(id.as_str().parse::<ResourceId>().unwrap(), id);
            assert_eq!(
                serde_json::from_str::<ResourceId>(&serde_json::to_string(&id).unwrap()).unwrap(),
                id
            );
        }
        assert_eq!(
            ResourceId::from_path("/tmp/a b").unwrap().as_str(),
            "file:///tmp/a%20b"
        );
        assert_eq!(
            "file://localhost/tmp/a"
                .parse::<ResourceId>()
                .unwrap()
                .as_str(),
            "file:///tmp/a"
        );
        assert!(
            "mem://notes/chapter"
                .parse::<ResourceId>()
                .unwrap()
                .to_file_path()
                .is_none()
        );
    }

    #[test]
    fn rejects_invalid_ids() {
        for uri in [
            "relative/path",
            "file:///tmp/a b",
            "file:///tmp/%",
            "file:///tmp/%xz",
            "file:///tmp/%00",
            "file://remote/tmp/a",
            "file:///tmp/a?x",
            "file:///tmp/a#x",
            "mem://x/\n",
        ] {
            assert!(uri.parse::<ResourceId>().is_err(), "{uri:?}");
            assert!(serde_json::from_value::<ResourceId>(serde_json::json!(uri)).is_err());
        }
        assert!(ResourceId::from_path("relative").is_err());
        let id = ResourceId::from_path("/tmp/a/../b").unwrap();
        assert_eq!(id.as_str(), "file:///tmp/b");
        assert_eq!(id.as_str().parse::<ResourceId>().unwrap(), id);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_path_round_trips() {
        use std::os::unix::ffi::OsStringExt;
        let path = PathBuf::from(std::ffi::OsString::from_vec(
            b"/tmp/invalid-\xff.md".to_vec(),
        ));
        let id = ResourceId::from_path(&path).unwrap();
        assert!(id.as_str().contains("%FF"));
        assert_eq!(id.to_file_path().unwrap(), path);
        assert_eq!(id.to_string().parse::<ResourceId>().unwrap(), id);
    }

    #[cfg(windows)]
    #[test]
    fn windows_drive_path_round_trips() {
        let path = Path::new(r"C:\notes\a b.md");
        let id = ResourceId::from_path(path).unwrap();
        assert_eq!(id.to_file_path().unwrap(), path);
        assert!(ResourceId::from_path(r"\\server\share\note.md").is_err());
    }

    #[test]
    fn image_region_serde() {
        let region = ResourceLocator::ImageRegion {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        };
        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["kind"], "image_region");
        assert_eq!(
            serde_json::from_value::<ResourceLocator>(json).unwrap(),
            region
        );
    }
}
