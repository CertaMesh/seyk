//! Embedding caches.
//!
//! A cache is an optimisation, not a corpus. It stores per-resource metadata,
//! per-segment locators and embedding vectors, never the source content.
//!
//! Vectors are keyed by `(profile, input_hash)`:
//! - the profile hashes everything that affects a vector (model, revision,
//!   precision, dimension, normalisation, prompts, segmenter version);
//! - the input hash covers the exact text (or bytes) sent to the model.
//!
//! So a vector is only ever reused for byte-identical input under an identical
//! profile. Size and mtime are only used to decide whether a file needs to be
//! re-read and re-hashed.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::embedder::ModelIdentity;
use crate::embedding::{from_bytes, to_bytes};
use crate::{Error, Modality, ResourceLocator, ResourceMetadata, Result};

/// Everything that determines a cached vector, besides the input itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheProfile {
    pub model: ModelIdentity,
    pub segmenter_version: String,
}

impl CacheProfile {
    /// Stable identifier derived from every field.
    pub fn id(&self) -> String {
        let json = serde_json::to_vec(self).expect("profile serialises");
        blake3::hash(&json).to_hex()[..24].to_string()
    }
}

/// A segment as remembered by the cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedSegment {
    pub locator: ResourceLocator,
    pub heading: Option<String>,
    /// Hash of the exact model input for this segment.
    pub input_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedResource {
    pub metadata: ResourceMetadata,
    pub segments: Vec<CachedSegment>,
}

/// A cached segment joined with its vector, ready to search.
#[derive(Debug, Clone)]
pub struct StoredVector {
    pub path: PathBuf,
    pub segment: CachedSegment,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileStats {
    pub profile_id: String,
    pub profile: Option<CacheProfile>,
    pub resources: u64,
    pub segments: u64,
    pub embeddings: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStats {
    /// Where the cache lives, when it is persistent.
    pub location: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub profiles: Vec<ProfileStats>,
}

/// Storage for resource metadata, segment locators and vectors.
pub trait EmbeddingCache: Send {
    fn register_profile(&mut self, id: &str, profile: &CacheProfile) -> Result<()>;

    /// Every cached resource for a profile, keyed by path. Segments are not loaded.
    fn resource_index(&self, profile: &str) -> Result<HashMap<PathBuf, ResourceMetadata>>;

    fn resource(&self, profile: &str, path: &Path) -> Result<Option<CachedResource>>;

    /// Replace a resource and its segments.
    fn put_resource(&mut self, profile: &str, path: &Path, resource: &CachedResource)
    -> Result<()>;

    /// Update only the stored mtime (content unchanged).
    fn touch_resource(
        &mut self,
        profile: &str,
        path: &Path,
        modified_ns: Option<i64>,
    ) -> Result<()>;

    fn remove_resources(&mut self, profile: &str, paths: &[PathBuf]) -> Result<()>;

    /// Which of `hashes` already have a vector.
    fn known_inputs(&self, profile: &str, hashes: &[String]) -> Result<HashSet<String>>;

    fn put_embeddings(&mut self, profile: &str, items: &[(String, Vec<f32>)]) -> Result<()>;

    /// All searchable vectors for the given paths.
    fn vectors_for(&self, profile: &str, paths: &HashSet<PathBuf>) -> Result<Vec<StoredVector>>;

    /// Drop vectors no segment refers to any more.
    fn prune_orphans(&mut self, profile: &str) -> Result<u64>;

    fn stats(&self) -> Result<CacheStats>;

    fn clear(&mut self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// In-memory cache (ephemeral searches and tests)
// ---------------------------------------------------------------------------

/// A cache that lives only as long as the value. Used for `--no-cache` searches.
#[derive(Debug, Default)]
pub struct MemoryCache {
    profiles: HashMap<String, CacheProfile>,
    // Bucketed by profile id, like the trait. Kept apart from `profiles`
    // because, unlike SQLite, writes may precede registration.
    resources: HashMap<String, HashMap<PathBuf, CachedResource>>,
    vectors: HashMap<String, HashMap<String, Vec<f32>>>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EmbeddingCache for MemoryCache {
    fn register_profile(&mut self, id: &str, profile: &CacheProfile) -> Result<()> {
        self.profiles.insert(id.to_string(), profile.clone());
        Ok(())
    }

    fn resource_index(&self, profile: &str) -> Result<HashMap<PathBuf, ResourceMetadata>> {
        Ok(self
            .resources
            .get(profile)
            .into_iter()
            .flatten()
            .map(|(path, r)| (path.clone(), r.metadata.clone()))
            .collect())
    }

    fn resource(&self, profile: &str, path: &Path) -> Result<Option<CachedResource>> {
        Ok(self
            .resources
            .get(profile)
            .and_then(|b| b.get(path))
            .cloned())
    }

    fn put_resource(
        &mut self,
        profile: &str,
        path: &Path,
        resource: &CachedResource,
    ) -> Result<()> {
        self.resources
            .entry(profile.to_string())
            .or_default()
            .insert(path.to_path_buf(), resource.clone());
        Ok(())
    }

    fn touch_resource(
        &mut self,
        profile: &str,
        path: &Path,
        modified_ns: Option<i64>,
    ) -> Result<()> {
        if let Some(r) = self
            .resources
            .get_mut(profile)
            .and_then(|b| b.get_mut(path))
        {
            r.metadata.modified_ns = modified_ns;
        }
        Ok(())
    }

    fn remove_resources(&mut self, profile: &str, paths: &[PathBuf]) -> Result<()> {
        if let Some(bucket) = self.resources.get_mut(profile) {
            for p in paths {
                bucket.remove(p);
            }
        }
        Ok(())
    }

    fn known_inputs(&self, profile: &str, hashes: &[String]) -> Result<HashSet<String>> {
        let Some(vectors) = self.vectors.get(profile) else {
            return Ok(HashSet::new());
        };
        Ok(hashes
            .iter()
            .filter(|h| vectors.contains_key(*h))
            .cloned()
            .collect())
    }

    fn put_embeddings(&mut self, profile: &str, items: &[(String, Vec<f32>)]) -> Result<()> {
        let vectors = self.vectors.entry(profile.to_string()).or_default();
        for (h, v) in items {
            vectors.insert(h.clone(), v.clone());
        }
        Ok(())
    }

    fn vectors_for(&self, profile: &str, paths: &HashSet<PathBuf>) -> Result<Vec<StoredVector>> {
        let (Some(resources), Some(vectors)) =
            (self.resources.get(profile), self.vectors.get(profile))
        else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (path, r) in resources {
            if !paths.contains(path) {
                continue;
            }
            for s in &r.segments {
                if let Some(v) = vectors.get(&s.input_hash) {
                    out.push(StoredVector {
                        path: path.clone(),
                        segment: s.clone(),
                        vector: v.clone(),
                    });
                }
            }
        }
        Ok(out)
    }

    fn prune_orphans(&mut self, profile: &str) -> Result<u64> {
        let Some(vectors) = self.vectors.get_mut(profile) else {
            return Ok(0);
        };
        let live: HashSet<&str> = self
            .resources
            .get(profile)
            .into_iter()
            .flat_map(|b| b.values())
            .flat_map(|r| r.segments.iter().map(|s| s.input_hash.as_str()))
            .collect();
        let before = vectors.len();
        vectors.retain(|h, _| live.contains(h.as_str()));
        Ok((before - vectors.len()) as u64)
    }

    fn stats(&self) -> Result<CacheStats> {
        let profiles = self
            .profiles
            .iter()
            .map(|(id, p)| {
                let resources = self.resources.get(id);
                ProfileStats {
                    profile_id: id.clone(),
                    profile: Some(p.clone()),
                    resources: resources.map_or(0, |b| b.len() as u64),
                    segments: resources
                        .into_iter()
                        .flat_map(|b| b.values())
                        .map(|r| r.segments.len() as u64)
                        .sum(),
                    embeddings: self.vectors.get(id).map_or(0, |b| b.len() as u64),
                }
            })
            .collect();
        Ok(CacheStats {
            location: None,
            size_bytes: None,
            profiles,
        })
    }

    fn clear(&mut self) -> Result<()> {
        *self = Self::default();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SQLite cache (default, persistent)
// ---------------------------------------------------------------------------

const SCHEMA_VERSION: &str = "1";

/// Persistent cache in a single SQLite file.
pub struct SqliteCache {
    conn: Connection,
    path: PathBuf,
}

impl SqliteCache {
    /// The default location: `<platform cache dir>/seyk/cache.sqlite3`.
    ///
    /// macOS: `~/Library/Caches/seyk/`, Linux: `$XDG_CACHE_HOME/seyk/`
    /// (usually `~/.cache/seyk/`).
    pub fn default_path() -> Result<PathBuf> {
        let base = dirs::cache_dir()
            .ok_or_else(|| Error::Config("no cache directory on this platform".into()))?;
        Ok(base.join("seyk").join("cache.sqlite3"))
    }

    pub fn open_default() -> Result<Self> {
        Self::open(Self::default_path()?)
    }

    /// Open or create a cache file. Embeddings can leak information about
    /// their source text, so the directory and file are private to the user.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
            restrict_permissions(dir, 0o700);
        }
        let conn = Connection::open(&path)?;
        restrict_permissions(&path, 0o600);
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        let mut cache = Self { conn, path };
        cache.migrate()?;
        Ok(cache)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn migrate(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )?;
        let version: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        if version.as_deref() != Some(SCHEMA_VERSION) {
            // It is a cache: on any schema change, start over rather than migrate.
            self.conn.execute_batch(
                "DROP TABLE IF EXISTS embeddings;
                 DROP TABLE IF EXISTS segments;
                 DROP TABLE IF EXISTS resources;
                 DROP TABLE IF EXISTS profiles;",
            )?;
        }
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS profiles (
                 id TEXT PRIMARY KEY,
                 profile TEXT NOT NULL,
                 dimension INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS resources (
                 profile TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                 path BLOB NOT NULL,
                 modality TEXT NOT NULL,
                 size_bytes INTEGER NOT NULL,
                 modified_ns INTEGER,
                 content_hash TEXT NOT NULL,
                 PRIMARY KEY (profile, path)
             );
             CREATE TABLE IF NOT EXISTS segments (
                 profile TEXT NOT NULL,
                 path BLOB NOT NULL,
                 ordinal INTEGER NOT NULL,
                 locator TEXT NOT NULL,
                 heading TEXT,
                 input_hash TEXT NOT NULL,
                 PRIMARY KEY (profile, path, ordinal),
                 FOREIGN KEY (profile, path) REFERENCES resources(profile, path) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS segments_by_input ON segments(profile, input_hash);
             CREATE TABLE IF NOT EXISTS embeddings (
                 profile TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                 input_hash TEXT NOT NULL,
                 vector BLOB NOT NULL,
                 PRIMARY KEY (profile, input_hash)
             );",
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION],
        )?;
        Ok(())
    }
}

#[cfg(unix)]
fn path_key(p: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    p.as_os_str().as_bytes().to_vec()
}

#[cfg(unix)]
fn key_path(b: Vec<u8>) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(std::ffi::OsString::from_vec(b))
}

#[cfg(not(unix))]
fn path_key(p: &Path) -> Vec<u8> {
    p.to_string_lossy().into_owned().into_bytes()
}

#[cfg(not(unix))]
fn key_path(b: Vec<u8>) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(&b).into_owned())
}

#[cfg(unix)]
fn restrict_permissions(p: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn restrict_permissions(_: &Path, _: u32) {}

fn parse_modality(s: &str) -> Modality {
    serde_json::from_value(serde_json::Value::String(s.to_string())).unwrap_or(Modality::Text)
}

fn json_err(e: serde_json::Error) -> Error {
    Error::Config(format!("corrupt cache entry: {e}"))
}

impl EmbeddingCache for SqliteCache {
    fn register_profile(&mut self, id: &str, profile: &CacheProfile) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO profiles (id, profile, dimension) VALUES (?1, ?2, ?3)",
            params![
                id,
                serde_json::to_string(profile).map_err(json_err)?,
                profile.model.dimension.get() as i64
            ],
        )?;
        Ok(())
    }

    fn resource_index(&self, profile: &str) -> Result<HashMap<PathBuf, ResourceMetadata>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT path, modality, size_bytes, modified_ns, content_hash FROM resources WHERE profile = ?1",
        )?;
        let rows = stmt.query_map([profile], |r| {
            Ok((
                key_path(r.get(0)?),
                ResourceMetadata {
                    modality: parse_modality(&r.get::<_, String>(1)?),
                    size_bytes: r.get::<_, i64>(2)? as u64,
                    modified_ns: r.get(3)?,
                    content_hash: r.get(4)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn resource(&self, profile: &str, path: &Path) -> Result<Option<CachedResource>> {
        let key = path_key(path);
        let metadata = self
            .conn
            .query_row(
                "SELECT modality, size_bytes, modified_ns, content_hash FROM resources WHERE profile = ?1 AND path = ?2",
                params![profile, key],
                |r| {
                    Ok(ResourceMetadata {
                        modality: parse_modality(&r.get::<_, String>(0)?),
                        size_bytes: r.get::<_, i64>(1)? as u64,
                        modified_ns: r.get(2)?,
                        content_hash: r.get(3)?,
                    })
                },
            )
            .optional()?;
        let Some(metadata) = metadata else {
            return Ok(None);
        };
        let mut stmt = self.conn.prepare_cached(
            "SELECT locator, heading, input_hash FROM segments WHERE profile = ?1 AND path = ?2 ORDER BY ordinal",
        )?;
        let segments = stmt
            .query_map(params![profile, key], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?
            .map(|row| {
                let (loc, heading, input_hash) = row?;
                Ok(CachedSegment {
                    locator: serde_json::from_str(&loc).map_err(json_err)?,
                    heading,
                    input_hash,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(CachedResource { metadata, segments }))
    }

    fn put_resource(
        &mut self,
        profile: &str,
        path: &Path,
        resource: &CachedResource,
    ) -> Result<()> {
        let key = path_key(path);
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM resources WHERE profile = ?1 AND path = ?2",
            params![profile, key],
        )?;
        let m = &resource.metadata;
        tx.execute(
            "INSERT INTO resources (profile, path, modality, size_bytes, modified_ns, content_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                profile,
                key,
                m.modality.to_string(),
                m.size_bytes as i64,
                m.modified_ns,
                m.content_hash
            ],
        )?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO segments (profile, path, ordinal, locator, heading, input_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (i, s) in resource.segments.iter().enumerate() {
                stmt.execute(params![
                    profile,
                    key,
                    i as i64,
                    serde_json::to_string(&s.locator).map_err(json_err)?,
                    s.heading,
                    s.input_hash
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn touch_resource(
        &mut self,
        profile: &str,
        path: &Path,
        modified_ns: Option<i64>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE resources SET modified_ns = ?3 WHERE profile = ?1 AND path = ?2",
            params![profile, path_key(path), modified_ns],
        )?;
        Ok(())
    }

    fn remove_resources(&mut self, profile: &str, paths: &[PathBuf]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt =
                tx.prepare_cached("DELETE FROM resources WHERE profile = ?1 AND path = ?2")?;
            for p in paths {
                stmt.execute(params![profile, path_key(p)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn known_inputs(&self, profile: &str, hashes: &[String]) -> Result<HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT 1 FROM embeddings WHERE profile = ?1 AND input_hash = ?2")?;
        let mut out = HashSet::new();
        for h in hashes {
            if stmt.exists(params![profile, h])? {
                out.insert(h.clone());
            }
        }
        Ok(out)
    }

    fn put_embeddings(&mut self, profile: &str, items: &[(String, Vec<f32>)]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO embeddings (profile, input_hash, vector) VALUES (?1, ?2, ?3)",
            )?;
            for (h, v) in items {
                stmt.execute(params![profile, h, to_bytes(v)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn vectors_for(&self, profile: &str, paths: &HashSet<PathBuf>) -> Result<Vec<StoredVector>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT s.path, s.locator, s.heading, s.input_hash, e.vector
             FROM segments s JOIN embeddings e ON e.profile = s.profile AND e.input_hash = s.input_hash
             WHERE s.profile = ?1
             ORDER BY s.path, s.ordinal",
        )?;
        let mut rows = stmt.query([profile])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let path = key_path(r.get(0)?);
            if !paths.contains(&path) {
                continue;
            }
            let locator: String = r.get(1)?;
            out.push(StoredVector {
                path,
                segment: CachedSegment {
                    locator: serde_json::from_str(&locator).map_err(json_err)?,
                    heading: r.get(2)?,
                    input_hash: r.get(3)?,
                },
                vector: from_bytes(&r.get::<_, Vec<u8>>(4)?),
            });
        }
        Ok(out)
    }

    fn prune_orphans(&mut self, profile: &str) -> Result<u64> {
        let n = self.conn.execute(
            "DELETE FROM embeddings WHERE profile = ?1 AND NOT EXISTS (
                 SELECT 1 FROM segments s WHERE s.profile = embeddings.profile AND s.input_hash = embeddings.input_hash)",
            [profile],
        )?;
        Ok(n as u64)
    }

    fn stats(&self) -> Result<CacheStats> {
        let mut stmt = self.conn.prepare(
            "SELECT p.id, p.profile,
                 (SELECT COUNT(*) FROM resources r WHERE r.profile = p.id),
                 (SELECT COUNT(*) FROM segments s WHERE s.profile = p.id),
                 (SELECT COUNT(*) FROM embeddings e WHERE e.profile = p.id)
             FROM profiles p ORDER BY p.id",
        )?;
        let profiles = stmt
            .query_map([], |r| {
                Ok(ProfileStats {
                    profile_id: r.get(0)?,
                    profile: serde_json::from_str(&r.get::<_, String>(1)?).ok(),
                    resources: r.get::<_, i64>(2)? as u64,
                    segments: r.get::<_, i64>(3)? as u64,
                    embeddings: r.get::<_, i64>(4)? as u64,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let size_bytes = ["", "-wal", "-shm"]
            .iter()
            .filter_map(|s| std::fs::metadata(format!("{}{s}", self.path.display())).ok())
            .map(|m| m.len())
            .sum();
        Ok(CacheStats {
            location: Some(self.path.clone()),
            size_bytes: Some(size_bytes),
            profiles,
        })
    }

    fn clear(&mut self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM embeddings; DELETE FROM segments; DELETE FROM resources; DELETE FROM profiles;")?;
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE); VACUUM;")?;
        Ok(())
    }
}
