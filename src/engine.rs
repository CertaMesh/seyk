//! The search engine: discovery, segmentation, lazy embedding, ranking.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cache::{CacheProfile, CachedResource, CachedSegment, EmbeddingCache, MemoryCache};
use crate::discovery::{DiscoveredResource, DiscoveryOptions, ResourceKind, discover};
use crate::embedder::{Embedder, ResourceInput};
use crate::index::{BruteForceIndex, VectorIndex};
use crate::segment::TextSegmenter;
use crate::{Error, ResourceId, ResourceLocator, ResourceMetadata, Result};

/// Bump when the way segments are turned into model inputs changes.
const INPUT_FORMAT_VERSION: u32 = 1;

/// One ranked match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub resource: ResourceId,
    pub locator: ResourceLocator,
    /// Cosine similarity in `[-1, 1]`. Only comparable within one model profile.
    pub score: f32,
    pub metadata: ResourceMetadata,
    /// Markdown heading path of the segment, when there is one.
    pub heading: Option<String>,
    /// A short excerpt of the matched segment (text resources only).
    pub preview: Option<String>,
}

/// A resource that was found but could not be searched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedResource {
    pub path: PathBuf,
    pub reason: String,
}

/// What happened while bringing the cache up to date.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncStats {
    pub resources_discovered: usize,
    /// Reused from cache without re-embedding.
    pub resources_unchanged: usize,
    /// New or changed since last seen.
    pub resources_updated: usize,
    pub skipped: Vec<SkippedResource>,
    /// Segments newly sent to the model.
    pub segments_embedded: usize,
    pub embed_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchStats {
    pub sync: SyncStats,
    pub segments_searched: usize,
    pub total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResults {
    pub query: String,
    pub hits: Vec<SearchHit>,
    pub stats: SearchStats,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexReport {
    pub sync: SyncStats,
    /// Cache entries removed because their files no longer exist under the indexed paths.
    pub resources_removed: usize,
    pub total_ms: u64,
}

/// Progress notifications for long operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProgressEvent {
    Discovered {
        resources: usize,
    },
    /// `total` segments need embedding; `done` so far.
    Embedding {
        done: usize,
        total: usize,
    },
}

pub type ProgressFn<'a> = &'a (dyn Fn(ProgressEvent) + Sync);

pub struct SearchEngineBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    cache: Option<Box<dyn EmbeddingCache>>,
    segmenter: TextSegmenter,
    batch_size: usize,
}

impl SearchEngineBuilder {
    pub fn embedder(mut self, embedder: impl Embedder + 'static) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    pub fn shared_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Where vectors are kept. Defaults to an in-memory cache, i.e. ephemeral search.
    pub fn cache(mut self, cache: impl EmbeddingCache + 'static) -> Self {
        self.cache = Some(Box::new(cache));
        self
    }

    pub fn segmenter(mut self, segmenter: TextSegmenter) -> Self {
        self.segmenter = segmenter;
        self
    }

    /// How many segments are sent to the embedder per call.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    pub fn build(self) -> Result<SearchEngine> {
        let embedder = self
            .embedder
            .ok_or_else(|| Error::Config("SearchEngine needs an embedder".into()))?;
        let profile = CacheProfile {
            model: embedder.identity().clone(),
            segmenter_version: format!(
                "{}+input-v{INPUT_FORMAT_VERSION}",
                self.segmenter.version()
            ),
        };
        let profile_id = profile.id();
        let mut cache = self.cache.unwrap_or_else(|| Box::new(MemoryCache::new()));
        cache.register_profile(&profile_id, &profile)?;
        Ok(SearchEngine {
            embedder,
            cache: Mutex::new(cache),
            segmenter: self.segmenter,
            profile,
            profile_id,
            batch_size: self.batch_size,
        })
    }
}

/// Semantic search over local files.
pub struct SearchEngine {
    embedder: Arc<dyn Embedder>,
    cache: Mutex<Box<dyn EmbeddingCache>>,
    segmenter: TextSegmenter,
    profile: CacheProfile,
    profile_id: String,
    batch_size: usize,
}

impl SearchEngine {
    pub fn builder() -> SearchEngineBuilder {
        SearchEngineBuilder {
            embedder: None,
            cache: None,
            segmenter: TextSegmenter::default(),
            batch_size: 16,
        }
    }

    pub fn profile(&self) -> &CacheProfile {
        &self.profile
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// Start a search request.
    pub fn search<'e>(&'e self, query: impl Into<String>) -> SearchRequest<'e> {
        SearchRequest {
            engine: self,
            query: query.into(),
            paths: Vec::new(),
            limit: 10,
            min_score: None,
            discovery: DiscoveryOptions::default(),
            preview_chars: Some(240),
            progress: None,
        }
    }

    /// Start an explicit indexing request (cache warming).
    pub fn index<'e>(&'e self) -> IndexRequest<'e> {
        IndexRequest {
            engine: self,
            paths: Vec::new(),
            discovery: DiscoveryOptions::default(),
            progress: None,
        }
    }

    /// Run a closure with exclusive access to the cache.
    pub fn with_cache<T>(&self, f: impl FnOnce(&mut dyn EmbeddingCache) -> Result<T>) -> Result<T> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| Error::Config("cache lock poisoned".into()))?;
        f(guard.as_mut())
    }

    /// Make sure every resource has up-to-date vectors in the cache.
    fn sync(
        &self,
        resources: &[DiscoveredResource],
        progress: Option<ProgressFn>,
    ) -> Result<SyncStats> {
        let mut stats = SyncStats {
            resources_discovered: resources.len(),
            ..Default::default()
        };
        let pid = self.profile_id.as_str();
        let known = self.with_cache(|c| c.resource_index(pid))?;

        let mut pending: Vec<Pending> = Vec::new();
        for r in resources {
            let modality = r.kind.modality();
            if !self.embedder.supports(modality) {
                stats.skipped.push(skip(
                    &r.path,
                    format!("{modality} search is not supported by this embedder"),
                ));
                continue;
            }
            let cached = known.get(&r.path);
            if let Some(m) = cached
                && r.modified_ns.is_some()
                && m.size_bytes == r.size_bytes
                && m.modified_ns == r.modified_ns
            {
                stats.resources_unchanged += 1;
                continue;
            }
            let bytes = match std::fs::read(&r.path) {
                Ok(b) => b,
                Err(e) => {
                    stats.skipped.push(skip(&r.path, e.to_string()));
                    continue;
                }
            };
            let content_hash = blake3::hash(&bytes).to_hex().to_string();
            if cached.is_some_and(|m| m.content_hash == content_hash) {
                self.with_cache(|c| c.touch_resource(pid, &r.path, r.modified_ns))?;
                stats.resources_unchanged += 1;
                continue;
            }
            let metadata = ResourceMetadata {
                modality,
                size_bytes: bytes.len() as u64,
                modified_ns: r.modified_ns,
                content_hash,
            };
            match self.prepare(r, bytes) {
                Ok(jobs) => pending.push(Pending {
                    path: r.path.clone(),
                    metadata,
                    jobs,
                }),
                Err(reason) => stats.skipped.push(skip(&r.path, reason)),
            }
        }

        // Only inputs the cache has never seen go to the model.
        let all_hashes: Vec<String> = pending
            .iter()
            .flat_map(|p| p.jobs.iter().map(|j| j.hash.clone()))
            .collect();
        let already = self.with_cache(|c| c.known_inputs(pid, &all_hashes))?;
        let mut queued: HashSet<&str> = HashSet::new();
        let mut work: Vec<(usize, &Job)> = Vec::new();
        for (ri, p) in pending.iter().enumerate() {
            for j in &p.jobs {
                if !already.contains(&j.hash) && queued.insert(j.hash.as_str()) {
                    work.push((ri, j));
                }
            }
        }

        let total = work.len();
        let started = Instant::now();
        let mut failed: HashMap<usize, String> = HashMap::new();
        let mut done = 0;
        let mut written = 0;
        if let Some(p) = progress {
            p(ProgressEvent::Embedding { done: 0, total });
        }
        for batch in work.chunks(self.batch_size) {
            let results = self.embed_batch(batch);
            let mut store = Vec::with_capacity(batch.len());
            for ((ri, job), res) in batch.iter().zip(results) {
                match res {
                    Ok(v) => store.push((job.hash.clone(), v)),
                    Err(e) => {
                        failed.entry(*ri).or_insert_with(|| e.to_string());
                    }
                }
            }
            self.with_cache(|c| c.put_embeddings(pid, &store))?;
            done += batch.len();
            stats.segments_embedded += store.len();
            if let Some(p) = progress {
                p(ProgressEvent::Embedding { done, total });
            }
            // Persist every resource whose segments are now all embedded, so an
            // interrupted run keeps its progress.
            let ready_upto = batch.last().map(|(ri, _)| *ri).unwrap_or(0);
            written = self.write_resources(&pending, written, ready_upto, &failed, &mut stats)?;
        }
        self.write_resources(&pending, written, pending.len(), &failed, &mut stats)?;
        stats.embed_ms = started.elapsed().as_millis() as u64;
        if stats.resources_updated > 0 {
            self.with_cache(|c| c.prune_orphans(pid))?;
        }
        Ok(stats)
    }

    /// Write `pending[from..upto]` to the cache. Returns the new `written` cursor.
    fn write_resources(
        &self,
        pending: &[Pending],
        from: usize,
        upto: usize,
        failed: &HashMap<usize, String>,
        stats: &mut SyncStats,
    ) -> Result<usize> {
        let pid = self.profile_id.as_str();
        for (i, p) in pending.iter().enumerate().take(upto).skip(from) {
            if let Some(reason) = failed.get(&i) {
                stats.skipped.push(skip(&p.path, reason.clone()));
                continue;
            }
            let resource = CachedResource {
                metadata: p.metadata.clone(),
                segments: p.jobs.iter().map(|j| j.segment.clone()).collect(),
            };
            self.with_cache(|c| c.put_resource(pid, &p.path, &resource))?;
            stats.resources_updated += 1;
        }
        Ok(upto.max(from))
    }

    /// Embed a batch; on failure retry one by one so a single bad input only skips its resource.
    fn embed_batch(&self, batch: &[(usize, &Job)]) -> Vec<Result<Vec<f32>>> {
        let inputs: Vec<ResourceInput> = batch.iter().map(|(_, j)| j.input.as_input()).collect();
        match self.embedder.embed_resources(&inputs) {
            Ok(vs) if vs.len() == inputs.len() => {
                vs.into_iter().map(|v| Ok(v.into_vec())).collect()
            }
            _ if inputs.len() > 1 => inputs
                .iter()
                .map(|i| {
                    self.embedder
                        .embed_resources(std::slice::from_ref(i))
                        .and_then(|mut v| {
                            v.pop()
                                .ok_or_else(|| Error::Embedding("no vector returned".into()))
                        })
                        .map(|e| e.into_vec())
                })
                .collect(),
            Ok(_) => vec![Err(Error::Embedding(
                "embedder returned the wrong number of vectors".into(),
            ))],
            Err(e) => vec![Err(e)],
        }
    }

    /// Turn a resource into embedding jobs. Errors are reasons to skip the resource.
    fn prepare(
        &self,
        r: &DiscoveredResource,
        bytes: Vec<u8>,
    ) -> std::result::Result<Vec<Job>, String> {
        match r.kind {
            ResourceKind::Text(format) => {
                if bytes.iter().take(8192).any(|&b| b == 0) {
                    return Err("looks like a binary file".into());
                }
                let text = String::from_utf8(bytes).map_err(|_| "not valid UTF-8".to_string())?;
                let stem = r.path.file_stem().map(|s| s.to_string_lossy().into_owned());
                Ok(self
                    .segmenter
                    .segment(&text, format)
                    .into_iter()
                    .map(|s| {
                        let title = segment_title(stem.as_deref(), s.heading.as_deref());
                        let hash = input_hash(&["text", title.as_deref().unwrap_or(""), &s.text]);
                        Job {
                            hash: hash.clone(),
                            segment: CachedSegment {
                                locator: ResourceLocator::Text {
                                    byte_start: s.byte_range.start as u64,
                                    byte_end: s.byte_range.end as u64,
                                    line_start: Some(s.line_start),
                                    line_end: Some(s.line_end),
                                },
                                heading: s.heading,
                                input_hash: hash,
                            },
                            input: JobInput::Text {
                                title,
                                text: s.text,
                            },
                        }
                    })
                    .collect())
            }
            ResourceKind::Image => {
                let hash = input_hash(&["image", &blake3::hash(&bytes).to_hex()]);
                Ok(vec![Job {
                    hash: hash.clone(),
                    segment: CachedSegment {
                        locator: ResourceLocator::WholeResource,
                        heading: None,
                        input_hash: hash,
                    },
                    input: JobInput::Image { bytes },
                }])
            }
        }
    }
}

struct Pending {
    path: PathBuf,
    metadata: ResourceMetadata,
    jobs: Vec<Job>,
}

struct Job {
    hash: String,
    segment: CachedSegment,
    input: JobInput,
}

enum JobInput {
    Text { title: Option<String>, text: String },
    Image { bytes: Vec<u8> },
}

impl JobInput {
    fn as_input(&self) -> ResourceInput<'_> {
        match self {
            JobInput::Text { title, text } => ResourceInput::Text {
                title: title.as_deref(),
                text,
            },
            JobInput::Image { bytes } => ResourceInput::Image { bytes },
        }
    }
}

fn skip(path: &Path, reason: impl Into<String>) -> SkippedResource {
    SkippedResource {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn input_hash(parts: &[&str]) -> String {
    let mut h = blake3::Hasher::new();
    for p in parts {
        h.update(&(p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    h.finalize().to_hex().to_string()
}

/// The document title sent to the model: the heading path, prefixed by the file
/// name unless the heading already starts with it.
fn segment_title(stem: Option<&str>, heading: Option<&str>) -> Option<String> {
    match (stem, heading) {
        (Some(s), Some(h)) if h.to_lowercase().starts_with(&s.to_lowercase()) => {
            Some(h.to_string())
        }
        (Some(s), Some(h)) => Some(format!("{s} > {h}")),
        (Some(s), None) => Some(s.to_string()),
        (None, h) => h.map(str::to_string),
    }
}

pub struct SearchRequest<'e> {
    engine: &'e SearchEngine,
    query: String,
    paths: Vec<PathBuf>,
    limit: usize,
    min_score: Option<f32>,
    discovery: DiscoveryOptions,
    preview_chars: Option<usize>,
    progress: Option<ProgressFn<'e>>,
}

impl<'e> SearchRequest<'e> {
    pub fn paths<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.paths
            .extend(paths.into_iter().map(|p| expand_tilde(p.as_ref())));
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    pub fn min_score(mut self, s: f32) -> Self {
        self.min_score = Some(s);
        self
    }

    pub fn include(mut self, glob: impl Into<String>) -> Self {
        self.discovery.include.push(glob.into());
        self
    }

    pub fn exclude(mut self, glob: impl Into<String>) -> Self {
        self.discovery.exclude.push(glob.into());
        self
    }

    pub fn discovery(mut self, opts: DiscoveryOptions) -> Self {
        self.discovery = opts;
        self
    }

    /// Maximum preview length in characters, or `None` to skip reading previews.
    pub fn previews(mut self, chars: Option<usize>) -> Self {
        self.preview_chars = chars;
        self
    }

    pub fn progress(mut self, f: ProgressFn<'e>) -> Self {
        self.progress = Some(f);
        self
    }

    pub fn execute(self) -> Result<SearchResults> {
        let started = Instant::now();
        let engine = self.engine;
        if self.paths.is_empty() {
            return Err(Error::Config("no paths to search".into()));
        }
        let resources = discover(&self.paths, &self.discovery)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let sync = engine.sync(&resources, self.progress)?;

        let query = engine.embedder.embed_query(&self.query)?;
        let dim = engine.embedder.dimension().get();
        if query.dim() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                actual: query.dim(),
            });
        }

        let pid = engine.profile_id.as_str();
        let wanted: HashSet<PathBuf> = resources.iter().map(|r| r.path.clone()).collect();
        let (stored, metadata) =
            engine.with_cache(|c| Ok((c.vectors_for(pid, &wanted)?, c.resource_index(pid)?)))?;

        let mut index = BruteForceIndex::with_capacity(dim, stored.len());
        let mut entries = Vec::with_capacity(stored.len());
        for v in stored {
            // A wrong-sized vector means a corrupt cache; never compare it.
            if v.vector.len() == dim {
                index.add(&v.vector);
                entries.push(v);
            }
        }
        let scored = index.search(
            query.as_slice(),
            self.limit,
            self.min_score.unwrap_or(f32::MIN),
        );

        let mut previews = PreviewReader::default();
        let hits = scored
            .into_iter()
            .filter_map(|s| {
                let e = &entries[s.id];
                let metadata = metadata.get(&e.path)?.clone();
                let preview = self
                    .preview_chars
                    .and_then(|n| previews.preview(&e.path, &e.segment.locator, &metadata, n));
                Some(SearchHit {
                    resource: ResourceId::from_path(&e.path),
                    locator: e.segment.locator.clone(),
                    score: s.score,
                    metadata,
                    heading: e.segment.heading.clone(),
                    preview,
                })
            })
            .collect();

        Ok(SearchResults {
            query: self.query,
            hits,
            stats: SearchStats {
                sync,
                segments_searched: entries.len(),
                total_ms: started.elapsed().as_millis() as u64,
            },
        })
    }
}

pub struct IndexRequest<'e> {
    engine: &'e SearchEngine,
    paths: Vec<PathBuf>,
    discovery: DiscoveryOptions,
    progress: Option<ProgressFn<'e>>,
}

impl<'e> IndexRequest<'e> {
    pub fn paths<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        self.paths
            .extend(paths.into_iter().map(|p| expand_tilde(p.as_ref())));
        self
    }

    pub fn discovery(mut self, opts: DiscoveryOptions) -> Self {
        self.discovery = opts;
        self
    }

    pub fn progress(mut self, f: ProgressFn<'e>) -> Self {
        self.progress = Some(f);
        self
    }

    /// Embed everything not yet cached and drop cache entries for files that
    /// were deleted under the indexed directories.
    pub fn execute(self) -> Result<IndexReport> {
        let started = Instant::now();
        let engine = self.engine;
        if self.paths.is_empty() {
            return Err(Error::Config("no paths to index".into()));
        }
        let resources = discover(&self.paths, &self.discovery)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let sync = engine.sync(&resources, self.progress)?;

        let pid = engine.profile_id.as_str();
        let roots: Vec<PathBuf> = self
            .paths
            .iter()
            .filter_map(|p| p.canonicalize().ok())
            .collect();
        let present: HashSet<&Path> = resources.iter().map(|r| r.path.as_path()).collect();
        let removed = engine.with_cache(|c| {
            let gone: Vec<PathBuf> = c
                .resource_index(pid)?
                .into_keys()
                .filter(|p| {
                    roots.iter().any(|r| p.starts_with(r))
                        && !present.contains(p.as_path())
                        && !p.exists()
                })
                .collect();
            c.remove_resources(pid, &gone)?;
            if !gone.is_empty() {
                c.prune_orphans(pid)?;
            }
            Ok(gone.len())
        })?;

        Ok(IndexReport {
            sync,
            resources_removed: removed,
            total_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// Reads previews, at most once per file per search.
#[derive(Default)]
struct PreviewReader {
    files: HashMap<PathBuf, Option<Vec<u8>>>,
}

impl PreviewReader {
    fn preview(
        &mut self,
        path: &Path,
        locator: &ResourceLocator,
        meta: &ResourceMetadata,
        max_chars: usize,
    ) -> Option<String> {
        let ResourceLocator::Text {
            byte_start,
            byte_end,
            ..
        } = locator
        else {
            return None;
        };
        let bytes = self
            .files
            .entry(path.to_path_buf())
            .or_insert_with(|| std::fs::read(path).ok())
            .as_ref()?;
        // If the file changed after it was embedded, the byte range may be wrong.
        if blake3::hash(bytes).to_hex().as_str() != meta.content_hash {
            return None;
        }
        let slice = bytes.get(*byte_start as usize..*byte_end as usize)?;
        Some(make_preview(&String::from_utf8_lossy(slice), max_chars))
    }
}

/// Collapse whitespace, drop leading heading lines and cut at a word boundary.
pub fn make_preview(text: &str, max_chars: usize) -> String {
    let body: Vec<&str> = text
        .lines()
        .skip_while(|l| l.trim_start().starts_with('#') || l.trim().is_empty())
        .collect();
    let source = if body.is_empty() {
        text.to_string()
    } else {
        body.join(" ")
    };
    let collapsed = source.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    let cut = cut.rsplit_once(' ').map(|(a, _)| a).unwrap_or(&cut);
    format!("{cut}…")
}

/// Expand a leading `~/` to the home directory (the shell does not do this inside quotes).
pub fn expand_tilde(p: &Path) -> PathBuf {
    match p.strip_prefix("~") {
        Ok(rest) => dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| p.to_path_buf()),
        Err(_) => p.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles() {
        assert_eq!(
            segment_title(Some("father"), None).as_deref(),
            Some("father")
        );
        assert_eq!(
            segment_title(Some("father"), Some("Father > Childhood")).as_deref(),
            Some("Father > Childhood")
        );
        assert_eq!(
            segment_title(Some("notes"), Some("Kitchen")).as_deref(),
            Some("notes > Kitchen")
        );
    }

    #[test]
    fn previews() {
        assert_eq!(
            make_preview("## Head\n\nSome   text\nhere.", 100),
            "Some text here."
        );
        assert_eq!(make_preview("alpha beta gamma delta", 12), "alpha beta…");
    }

    #[test]
    fn input_hash_is_unambiguous() {
        assert_ne!(
            input_hash(&["text", "ab", "c"]),
            input_hash(&["text", "a", "bc"])
        );
    }
}
