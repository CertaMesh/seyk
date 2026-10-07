//! Embedding caches.
//!
//! A cache is an optimisation, not a corpus. It stores per-resource metadata,
//! per-segment locators, embedding vectors and a keyword index of each
//! segment's words, never a readable copy of the source content. (The keyword
//! index retains word frequencies, not source word order, and is still sensitive.)
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
use crate::lexical::{FIELDS, LexicalDoc, LexicalQuery, order_free_text};
use crate::{
    Error, Modality, ResourceId, ResourceLocator, ResourceMetadata, ResourceVersion, Result,
};

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
    /// Text for the keyword index. Write-only: it is indexed on `put_resource`
    /// and always `None` when read back, so the cache holds no readable copy.
    #[serde(skip)]
    pub lexical_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedResource {
    pub metadata: ResourceMetadata,
    pub version: ResourceVersion,
    /// Preparation descriptor hash; absent for migrated legacy entries.
    pub descriptor_hash: Option<String>,
    pub segments: Vec<CachedSegment>,
}

/// Resource metadata and the provider token used to index it.
///
/// Unstable: may change before 1.0 alongside the provider and cache contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedResourceState {
    pub metadata: ResourceMetadata,
    pub version: ResourceVersion,
    /// Preparation descriptor hash; absent for migrated legacy entries.
    pub descriptor_hash: Option<String>,
}

/// A cached segment joined with its vector, ready to search.
#[derive(Debug, Clone)]
pub struct StoredVector {
    pub resource: ResourceId,
    /// Position of the segment within its resource.
    pub ordinal: usize,
    pub segment: CachedSegment,
    pub vector: Vec<f32>,
}

/// A segment matching a keyword query.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalHit {
    pub resource: ResourceId,
    pub ordinal: usize,
    /// BM25 relevance, higher is better. Only comparable within one query.
    pub score: f32,
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
///
/// Unstable: exposes ResourceVersion and CachedResourceState, which may change
/// before 1.0; filesystem is the only supported production provider.
pub trait EmbeddingCache: Send {
    fn register_profile(&mut self, id: &str, profile: &CacheProfile) -> Result<()>;

    /// Every cached resource for a profile, keyed by identity. Segments are not loaded.
    fn resource_index(&self, profile: &str) -> Result<HashMap<ResourceId, CachedResourceState>>;

    fn resource(&self, profile: &str, path: &ResourceId) -> Result<Option<CachedResource>>;

    /// Replace a resource and its segments.
    fn put_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        resource: &CachedResource,
    ) -> Result<()>;

    /// Update metadata and version when content is unchanged.
    fn touch_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        metadata: &ResourceMetadata,
        version: &ResourceVersion,
    ) -> Result<()>;

    fn remove_resources(&mut self, profile: &str, paths: &[ResourceId]) -> Result<()>;

    /// Which of `hashes` already have a vector.
    fn known_inputs(&self, profile: &str, hashes: &[String]) -> Result<HashSet<String>>;

    fn put_embeddings(&mut self, profile: &str, items: &[(String, Vec<f32>)]) -> Result<()>;

    /// All searchable vectors for the given paths.
    fn vectors_for(&self, profile: &str, paths: &HashSet<ResourceId>) -> Result<Vec<StoredVector>>;

    /// Segments of `paths` matching `query` by keyword, best first, at most `limit`.
    /// Equal scores are ordered by path bytes, then segment ordinal.
    fn lexical_search(
        &self,
        profile: &str,
        query: &LexicalQuery,
        paths: &HashSet<ResourceId>,
        limit: usize,
    ) -> Result<Vec<LexicalHit>>;

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
    resources: HashMap<String, HashMap<ResourceId, CachedResource>>,
    vectors: HashMap<String, HashMap<String, Vec<f32>>>,
    /// Keyword index per profile and path, one document per segment.
    lexical: HashMap<String, HashMap<ResourceId, Vec<LexicalDoc>>>,
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

    fn resource_index(&self, profile: &str) -> Result<HashMap<ResourceId, CachedResourceState>> {
        Ok(self
            .resources
            .get(profile)
            .into_iter()
            .flatten()
            .map(|(path, r)| {
                (
                    path.clone(),
                    CachedResourceState {
                        metadata: r.metadata.clone(),
                        version: r.version.clone(),
                        descriptor_hash: r.descriptor_hash.clone(),
                    },
                )
            })
            .collect())
    }

    fn resource(&self, profile: &str, path: &ResourceId) -> Result<Option<CachedResource>> {
        Ok(self
            .resources
            .get(profile)
            .and_then(|b| b.get(path))
            .cloned())
    }

    fn put_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        resource: &CachedResource,
    ) -> Result<()> {
        let docs = lexical_docs(path, resource);
        // Like SQLite, keep no readable text: it is indexed, then dropped.
        let mut stored = resource.clone();
        for s in &mut stored.segments {
            s.lexical_text = None;
        }
        self.resources
            .entry(profile.to_string())
            .or_default()
            .insert(path.clone(), stored);
        self.lexical
            .entry(profile.to_string())
            .or_default()
            .insert(path.clone(), docs);
        Ok(())
    }

    fn touch_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        metadata: &ResourceMetadata,
        version: &ResourceVersion,
    ) -> Result<()> {
        if let Some(r) = self
            .resources
            .get_mut(profile)
            .and_then(|b| b.get_mut(path))
        {
            r.metadata = metadata.clone();
            r.version = version.clone();
        }
        Ok(())
    }

    fn remove_resources(&mut self, profile: &str, paths: &[ResourceId]) -> Result<()> {
        if let Some(bucket) = self.resources.get_mut(profile) {
            for p in paths {
                bucket.remove(p);
            }
        }
        if let Some(bucket) = self.lexical.get_mut(profile) {
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

    fn vectors_for(&self, profile: &str, paths: &HashSet<ResourceId>) -> Result<Vec<StoredVector>> {
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
            for (ordinal, s) in r.segments.iter().enumerate() {
                if let Some(v) = vectors.get(&s.input_hash) {
                    out.push(StoredVector {
                        resource: path.clone(),
                        ordinal,
                        segment: s.clone(),
                        vector: v.clone(),
                    });
                }
            }
        }
        Ok(out)
    }

    fn lexical_search(
        &self,
        profile: &str,
        query: &LexicalQuery,
        paths: &HashSet<ResourceId>,
        limit: usize,
    ) -> Result<Vec<LexicalHit>> {
        let Some(bucket) = self.lexical.get(profile) else {
            return Ok(Vec::new());
        };
        // Score against the whole profile, as FTS5 does, so document frequencies
        // do not depend on which paths were asked for.
        let docs = bucket
            .iter()
            .flat_map(|(path, docs)| docs.iter().enumerate().map(move |(i, d)| ((path, i), d)));
        let mut scored = crate::lexical::bm25(docs, query);
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| resource_key(a.0.0).cmp(&resource_key(b.0.0)))
                .then_with(|| a.0.1.cmp(&b.0.1))
        });
        Ok(scored
            .into_iter()
            .filter(|((path, _), _)| paths.contains(*path))
            .take(limit)
            .map(|((path, ordinal), score)| LexicalHit {
                resource: path.clone(),
                ordinal,
                score: score as f32,
            })
            .collect())
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

/// One keyword-index document per segment: file name, path, heading and text.
fn lexical_docs(path: &ResourceId, resource: &CachedResource) -> Vec<LexicalDoc> {
    let (name, full) = path.labels();
    resource
        .segments
        .iter()
        .map(|s| {
            LexicalDoc::new(
                &name,
                &full,
                s.heading.as_deref(),
                s.lexical_text.as_deref().unwrap_or(""),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SQLite cache (default, persistent)
// ---------------------------------------------------------------------------

const SCHEMA_VERSION: &str = "4";

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
        let tx = self.conn.transaction()?;
        if matches!(version.as_deref(), Some("1" | "2")) {
            // v3 removes source word order. Keep segment references so orphan
            // pruning cannot remove vectors for roots that have not synced yet.
            // v1 already has the same resource/segment columns, but no FTS table.
            let has_fts: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'segments_fts')",
                [],
                |r| r.get(0),
            )?;
            if has_fts {
                // A plain DELETE only tombstones contentless rows: the old
                // tokens and positions stay in segments_fts_data. delete-all
                // rebuilds the index empty.
                tx.execute_batch("INSERT INTO segments_fts(segments_fts) VALUES ('delete-all');")?;
            }
            // Invalidate both fast paths: every resource must be re-read and
            // re-segmented, reusing its vectors, on its next normal sync.
            tx.execute_batch("UPDATE resources SET modified_ns = NULL, content_hash = '';")?;
        } else if !matches!(version.as_deref(), Some("3") | Some(SCHEMA_VERSION)) {
            // It is a cache: on any other schema change, start over.
            tx.execute_batch(
                "DROP TABLE IF EXISTS segments_fts;
                 DROP TABLE IF EXISTS embeddings;
                 DROP TABLE IF EXISTS segments;
                 DROP TABLE IF EXISTS resources;
                 DROP TABLE IF EXISTS profiles;",
            )?;
        }
        tx.execute_batch(
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
                 version TEXT,
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
             );
             -- Sorted input: full detail preserves frequencies, not source order.
             -- Contentless: stores the words' index, not the text. Row ids are
             -- segments.rowid, so rows are deleted explicitly with their segments.
             CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                 name, path, heading, body,
                 content = '', contentless_delete = 1,
                 tokenize = 'unicode61 remove_diacritics 2'
             );",
        )?;
        // Add the provider token and preparation fingerprint: keys, segment rowids, FTS and vectors
        // remain byte-for-byte intact when upgrading an existing v3 cache.
        let has_version: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('resources') WHERE name = 'version')",
            [],
            |r| r.get(0),
        )?;
        if !has_version {
            tx.execute_batch("ALTER TABLE resources ADD COLUMN version TEXT;")?;
        }
        let has_descriptor: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('resources') WHERE name = 'descriptor_hash')",
            [], |r| r.get(0),
        )?;
        if !has_descriptor {
            tx.execute_batch("ALTER TABLE resources ADD COLUMN descriptor_hash TEXT;")?;
        }
        if matches!(version.as_deref(), Some("1" | "2" | "3")) {
            tx.execute_batch(
                r#"UPDATE resources SET version = CASE
                WHEN modified_ns IS NOT NULL AND content_hash != ''
                THEN '"fs:' || size_bytes || ':' || modified_ns || '"'
                ELSE 'null' END;"#,
            )?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION],
        )?;
        tx.commit()?;
        Ok(())
    }
}

// File keys retain the legacy encoding and sort order. NUL cannot occur in
// a valid file path, so it separates non-file identities without collisions.
pub(crate) fn resource_key(id: &ResourceId) -> Vec<u8> {
    match id.to_file_path() {
        Some(path) => path_key(&path),
        None => std::iter::once(0).chain(id.as_str().bytes()).collect(),
    }
}

fn key_resource(bytes: Vec<u8>) -> rusqlite::Result<ResourceId> {
    let result = if let Some(uri) = bytes.strip_prefix(&[0]) {
        std::str::from_utf8(uri)
            .map_err(|e| Error::Config(e.to_string()))
            .and_then(str::parse)
    } else {
        ResourceId::from_path(key_path(bytes))
    };
    result.map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(e))
    })
}

fn decode_version(value: Option<String>) -> rusqlite::Result<ResourceVersion> {
    value
        .map(|v| {
            serde_json::from_str(&v).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

#[cfg(unix)]
pub(crate) fn path_key(p: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    p.as_os_str().as_bytes().to_vec()
}

#[cfg(unix)]
fn key_path(b: Vec<u8>) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(std::ffi::OsString::from_vec(b))
}

#[cfg(windows)]
pub(crate) fn path_key(p: &Path) -> Vec<u8> {
    // Discovery has always stored canonical Windows paths, including the
    // verbatim drive prefix. URI decoding omits that prefix; restore it so a
    // pre-v4 cache continues to use exactly the same keys.
    use std::path::{Component, Prefix};
    let raw = p.to_string_lossy();
    match p.components().next() {
        Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Disk(_)) => {
            format!(r"\\?\{raw}").into_bytes()
        }
        _ => raw.into_owned().into_bytes(),
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn path_key(p: &Path) -> Vec<u8> {
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

/// Remove a resource's keyword rows. Contentless FTS rows are not covered by
/// the foreign-key cascade, so they go first, while their segment rows exist.
fn delete_fts_rows(tx: &rusqlite::Transaction, profile: &str, key: &[u8]) -> Result<()> {
    tx.execute(
        "DELETE FROM segments_fts WHERE rowid IN
             (SELECT rowid FROM segments WHERE profile = ?1 AND path = ?2)",
        params![profile, key],
    )?;
    Ok(())
}

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

    fn resource_index(&self, profile: &str) -> Result<HashMap<ResourceId, CachedResourceState>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT path, modality, size_bytes, modified_ns, content_hash, version, descriptor_hash FROM resources WHERE profile = ?1",
        )?;
        let rows = stmt.query_map([profile], |r| {
            let id = key_resource(r.get(0)?)?;
            let metadata = ResourceMetadata {
                modality: parse_modality(&r.get::<_, String>(1)?),
                size_bytes: r.get::<_, i64>(2)? as u64,
                modified_ns: r.get(3)?,
                content_hash: r.get(4)?,
            };
            let version = decode_version(r.get(5)?)?;
            Ok((
                id,
                CachedResourceState {
                    metadata,
                    version,
                    descriptor_hash: r.get(6)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn resource(&self, profile: &str, path: &ResourceId) -> Result<Option<CachedResource>> {
        let key = resource_key(path);
        let metadata = self
            .conn
            .query_row(
                "SELECT modality, size_bytes, modified_ns, content_hash, version, descriptor_hash FROM resources WHERE profile = ?1 AND path = ?2",
                params![profile, key],
                |r| {
                    Ok((ResourceMetadata {
                        modality: parse_modality(&r.get::<_, String>(0)?),
                        size_bytes: r.get::<_, i64>(1)? as u64,
                        modified_ns: r.get(2)?,
                        content_hash: r.get(3)?,
                    }, decode_version(r.get(4)?)?, r.get::<_, Option<String>>(5)?))
                },
            )
            .optional()?;
        let Some((metadata, version, descriptor_hash)) = metadata else {
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
                    lexical_text: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(CachedResource {
            metadata,
            version,
            descriptor_hash,
            segments,
        }))
    }

    fn put_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        resource: &CachedResource,
    ) -> Result<()> {
        let key = resource_key(path);
        let tx = self.conn.transaction()?;
        delete_fts_rows(&tx, profile, &key)?;
        tx.execute(
            "DELETE FROM resources WHERE profile = ?1 AND path = ?2",
            params![profile, key],
        )?;
        let m = &resource.metadata;
        tx.execute(
            "INSERT INTO resources (profile, path, modality, size_bytes, modified_ns, content_hash, version, descriptor_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                profile,
                key,
                m.modality.to_string(),
                m.size_bytes as i64,
                m.modified_ns,
                m.content_hash,
                serde_json::to_string(&resource.version).map_err(json_err)?,
                resource.descriptor_hash
            ],
        )?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO segments (profile, path, ordinal, locator, heading, input_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut fts = tx.prepare_cached(
                "INSERT INTO segments_fts (rowid, name, path, heading, body) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let (name, full) = path.labels();
            let name = order_free_text(&name);
            let full = order_free_text(&full);
            for (i, s) in resource.segments.iter().enumerate() {
                stmt.execute(params![
                    profile,
                    key,
                    i as i64,
                    serde_json::to_string(&s.locator).map_err(json_err)?,
                    s.heading,
                    s.input_hash
                ])?;
                fts.execute(params![
                    tx.last_insert_rowid(),
                    name,
                    full,
                    order_free_text(s.heading.as_deref().unwrap_or("")),
                    order_free_text(s.lexical_text.as_deref().unwrap_or(""))
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn touch_resource(
        &mut self,
        profile: &str,
        path: &ResourceId,
        metadata: &ResourceMetadata,
        version: &ResourceVersion,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE resources SET modified_ns = ?3, size_bytes = ?4, version = ?5, content_hash = ?6, modality = ?7 WHERE profile = ?1 AND path = ?2",
            params![profile, resource_key(path), metadata.modified_ns, metadata.size_bytes as i64, serde_json::to_string(version).map_err(json_err)?, metadata.content_hash, metadata.modality.to_string()],
        )?;
        Ok(())
    }

    fn remove_resources(&mut self, profile: &str, paths: &[ResourceId]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt =
                tx.prepare_cached("DELETE FROM resources WHERE profile = ?1 AND path = ?2")?;
            for p in paths {
                let key = resource_key(p);
                delete_fts_rows(&tx, profile, &key)?;
                stmt.execute(params![profile, key])?;
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

    fn vectors_for(&self, profile: &str, paths: &HashSet<ResourceId>) -> Result<Vec<StoredVector>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT s.path, s.locator, s.heading, s.input_hash, e.vector, s.ordinal
             FROM segments s JOIN embeddings e ON e.profile = s.profile AND e.input_hash = s.input_hash
             WHERE s.profile = ?1
             ORDER BY s.path, s.ordinal",
        )?;
        let mut rows = stmt.query([profile])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let path = key_resource(r.get(0)?)?;
            if !paths.contains(&path) {
                continue;
            }
            let locator: String = r.get(1)?;
            out.push(StoredVector {
                resource: path,
                ordinal: r.get::<_, i64>(5)? as usize,
                segment: CachedSegment {
                    locator: serde_json::from_str(&locator).map_err(json_err)?,
                    heading: r.get(2)?,
                    input_hash: r.get(3)?,
                    lexical_text: None,
                },
                vector: from_bytes(&r.get::<_, Vec<u8>>(4)?),
            });
        }
        Ok(out)
    }

    fn lexical_search(
        &self,
        profile: &str,
        query: &LexicalQuery,
        paths: &HashSet<ResourceId>,
        limit: usize,
    ) -> Result<Vec<LexicalHit>> {
        let Some(expr) = query.fts5() else {
            return Ok(Vec::new());
        };
        let weights = FIELDS.map(|(_, w)| w.to_string()).join(", ");
        // bm25() is lower-is-better; ties use the stored path bytes and ordinal
        // before filtering requested paths and truncating the candidate list.
        let sql = format!(
            "SELECT s.path, s.ordinal, -bm25(segments_fts, {weights})
             FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid
             WHERE segments_fts MATCH ?1 AND s.profile = ?2
             ORDER BY bm25(segments_fts, {weights}), s.path, s.ordinal"
        );
        let mut stmt = self.conn.prepare_cached(&sql)?;
        let mut rows = stmt.query(params![expr, profile])?;
        let mut out = Vec::new();
        while out.len() < limit
            && let Some(r) = rows.next()?
        {
            let path = key_resource(r.get(0)?)?;
            if paths.contains(&path) {
                out.push(LexicalHit {
                    resource: path,
                    ordinal: r.get::<_, i64>(1)? as usize,
                    score: r.get::<_, f64>(2)? as f32,
                });
            }
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
            .execute_batch("INSERT INTO segments_fts(segments_fts) VALUES ('delete-all'); DELETE FROM embeddings; DELETE FROM segments; DELETE FROM resources; DELETE FROM profiles;")?;
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE); VACUUM;")?;
        Ok(())
    }
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    #[test]
    fn file_keys_keep_the_legacy_canonical_drive_prefix() {
        let id = ResourceId::from_path(r"C:\notes\chapter.md").unwrap();
        let legacy = br"\\?\C:\notes\chapter.md".to_vec();
        assert_eq!(resource_key(&id), legacy);
        assert_eq!(key_resource(legacy).unwrap(), id);
    }
}
