//! The search engine: discovery, segmentation, lazy embedding, ranking.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cache::{
    CacheProfile, CachedSegment, EmbeddingCache, MemoryCache, ResourceWrite, SegmentWrite,
};
use crate::discovery::{DiscoveryOptions, ResourceKind};
use crate::embedder::{Embedder, EmbeddingRequest, EmbeddingTask, ResourceInput};
use crate::extractor::{
    ExtractedContent, Extractor, ResourceBytes, TextExtractor, extract_resource, text_range,
    unsupported,
};
use crate::index::{BruteForceIndex, Scored, VectorIndex};
use crate::lexical::LexicalQuery;
use crate::segment::{TextFormat, TextSegmenter};
use crate::{Embedding, Error, ResourceId, ResourceLocator, ResourceMetadata, Result};
use crate::{FsProvider, ProvidedResource, ResourceProvider, ResourceScope, ResourceVersion};

/// Bump when the way segments are turned into model inputs changes.
const INPUT_FORMAT_VERSION: u32 = 2;

/// How results are found and ranked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// Meaning, keywords and file names, fused by rank. The default.
    #[default]
    Hybrid,
    /// Embedding similarity only; `score` is the cosine similarity.
    Semantic,
    /// Keywords and file names only. The query is never embedded.
    Lexical,
}

impl std::str::FromStr for SearchMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "hybrid" => Ok(Self::Hybrid),
            "semantic" => Ok(Self::Semantic),
            "lexical" | "keyword" => Ok(Self::Lexical),
            other => Err(Error::Config(format!(
                "unknown search mode {other:?}: use hybrid, semantic or lexical"
            ))),
        }
    }
}

/// Which signals found a hit, with their raw scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MatchSignals {
    /// Cosine similarity of the best-matching phrasing, when it ranked.
    pub semantic: Option<f32>,
    /// BM25 keyword relevance, when it ranked. Higher is better.
    pub lexical: Option<f32>,
    /// The query named this file (`father.md`, `father`, `notes/father.md`).
    pub filename: bool,
}

/// One ranked match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub resource: ResourceId,
    pub locator: ResourceLocator,
    /// Relevance. In hybrid and lexical mode a rank fusion in `(0, 1]`, where 1
    /// means first on every signal that found anything; in semantic mode the
    /// cosine similarity. Only comparable within one search.
    pub score: f32,
    /// What found this hit. Use it to tell a meaning match from a name match.
    pub signals: MatchSignals,
    pub metadata: ResourceMetadata,
    /// Markdown heading path of the segment, when there is one.
    pub heading: Option<String>,
    /// A short excerpt of the matched segment (text resources only).
    pub preview: Option<String>,
    /// Index into [`SearchResults::queries`] of the query that scored best here.
    /// Meaningful only when [`MatchSignals::semantic`] is present.
    pub query_index: usize,
}

/// A resource that was found but could not be searched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedResource {
    pub resource: ResourceId,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::resource::serialize_optional_path"
    )]
    pub path: Option<PathBuf>,
    pub reason: String,
}

/// A page omitted from a resource that was otherwise indexed successfully.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageDiagnostic {
    pub resource: ResourceId,
    pub page: u32,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub page_diagnostics: Vec<PageDiagnostic>,
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
    /// Every phrasing that was searched, in the order given.
    pub queries: Vec<String>,
    pub mode: SearchMode,
    pub hits: Vec<SearchHit>,
    pub stats: SearchStats,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexReport {
    pub sync: SyncStats,
    /// Cache entries removed after the provider confirms in-scope deletion.
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

/// Called without the cache mutex held. During embedding, the indexing lease
/// is held: callbacks may inspect the cache or search with `sync(false)`, but
/// must not synchronously start another indexing pass on the same engine.
pub type ProgressFn<'a> = &'a (dyn Fn(ProgressEvent) + Sync);

pub struct SearchEngineBuilder {
    provider: Arc<dyn ResourceProvider>,
    extractors: Vec<Arc<dyn Extractor>>,
    #[cfg(feature = "audio")]
    audio_extractor: Option<crate::audio::AudioExtractor>,
    embedder: Option<Arc<dyn Embedder>>,
    cache: Option<Box<dyn EmbeddingCache>>,
    segmenter: TextSegmenter,
    batch_size: usize,
    max_audio_duration_secs: u64,
}

impl SearchEngineBuilder {
    /// Explicit audio extraction only; never registered with preview dispatch.
    #[cfg(feature = "audio")]
    pub fn audio_extractor(mut self, extractor: crate::audio::AudioExtractor) -> Self {
        self.audio_extractor = Some(extractor);
        self
    }

    /// Register an extractor before the built-in text fallback. Custom
    /// extractors are tried in registration order; errors never fall through.
    pub fn extractor(mut self, extractor: impl Extractor + 'static) -> Self {
        self.extractors.push(Arc::new(extractor));
        self
    }

    /// Resource access; defaults to the local filesystem provider.
    pub fn provider(mut self, provider: impl ResourceProvider + 'static) -> Self {
        self.provider = Arc::new(provider);
        self
    }

    pub fn shared_provider(mut self, provider: Arc<dyn ResourceProvider>) -> Self {
        self.provider = provider;
        self
    }

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

    /// Inclusive canonical audio duration limit; independent of vector identity.
    pub fn max_audio_duration_secs(mut self, seconds: u64) -> Self {
        self.max_audio_duration_secs = seconds;
        self
    }

    pub fn build(mut self) -> Result<SearchEngine> {
        crate::discovery::audio_byte_limit(self.max_audio_duration_secs)?;
        #[cfg(feature = "pdf")]
        self.extractors.push(Arc::new(crate::PdfTextExtractor));
        self.extractors.push(Arc::new(TextExtractor));
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
            provider: self.provider,
            extractors: self.extractors,
            #[cfg(feature = "audio")]
            audio_extractor: self.audio_extractor,
            embedder,
            cache: Mutex::new(cache),
            indexing: Mutex::new(()),
            segmenter: self.segmenter,
            profile,
            profile_id,
            batch_size: self.batch_size,
            max_audio_duration_secs: self.max_audio_duration_secs,
        })
    }
}

/// Search over provider resources; local files are the default.
///
/// Indexing passes on one engine are serialized through resource commits and
/// orphan pruning. Separate engines, including two processes sharing one SQLite
/// cache, are not covered by this engine-local lease. Cached searches do not
/// acquire it.
pub struct SearchEngine {
    provider: Arc<dyn ResourceProvider>,
    extractors: Vec<Arc<dyn Extractor>>,
    #[cfg(feature = "audio")]
    audio_extractor: Option<crate::audio::AudioExtractor>,
    embedder: Arc<dyn Embedder>,
    cache: Mutex<Box<dyn EmbeddingCache>>,
    /// Acquire before `cache`, never while holding it. Covers pending vectors
    /// until their resource references are committed and pruning is complete.
    indexing: Mutex<()>,
    segmenter: TextSegmenter,
    profile: CacheProfile,
    profile_id: String,
    batch_size: usize,
    max_audio_duration_secs: u64,
}

impl SearchEngine {
    pub fn builder() -> SearchEngineBuilder {
        SearchEngineBuilder {
            provider: Arc::new(FsProvider),
            extractors: Vec::new(),
            #[cfg(feature = "audio")]
            audio_extractor: None,
            embedder: None,
            cache: None,
            segmenter: TextSegmenter::default(),
            batch_size: 16,
            max_audio_duration_secs: 7200,
        }
    }

    /// Extract readable evidence from a hit, verifying its full-resource hash
    /// before any extractor receives bytes. Only explicit audio extraction may run STT.
    pub fn extract(&self, hit: &SearchHit) -> Result<ExtractedContent> {
        #[cfg(feature = "audio")]
        if hit.metadata.modality == crate::Modality::Audio {
            if let Some(extractor) = &self.audio_extractor {
                return extractor.extract(
                    self.provider.as_ref(),
                    &hit.resource,
                    &hit.locator,
                    &hit.metadata,
                    self.max_audio_duration_secs,
                );
            }
            self.extract_audio_samples(hit)?;
            return Err(unsupported(&hit.resource, &hit.locator, &hit.metadata));
        }
        let extractor = self
            .extractors
            .iter()
            .find(|e| e.supports(&hit.resource, &hit.locator, &hit.metadata))
            .ok_or_else(|| unsupported(&hit.resource, &hit.locator, &hit.metadata))?;
        extract_resource(
            self.provider.as_ref(),
            extractor.as_ref(),
            &hit.resource,
            &hit.locator,
            &hit.metadata,
        )
    }

    /// Verified selected audio samples, without STT. At most 30 seconds per slice.
    #[cfg(feature = "audio")]
    pub fn extract_audio_samples(&self, hit: &SearchHit) -> Result<Vec<f32>> {
        crate::audio::extract_samples(
            self.provider.as_ref(),
            &hit.resource,
            &hit.locator,
            &hit.metadata,
            self.max_audio_duration_secs,
        )
    }

    /// Sync and cached-only search must agree on the prepared audio identity.
    fn audio_descriptor_hash(&self, resource: &ProvidedResource) -> String {
        #[cfg(feature = "audio")]
        {
            input_hash(&[
                &resource.fingerprint(),
                crate::audio::PREPARATION,
                self.embedder.input_identity(crate::Modality::Audio),
            ])
        }
        #[cfg(not(feature = "audio"))]
        {
            resource.fingerprint()
        }
    }

    fn audio_eligible(&self, id: &ResourceId) -> Result<bool> {
        let limit = self.max_audio_duration_secs * 1000;
        self.with_cache(|c| {
            let Some(r) = c.resource(&self.profile_id, id)? else {return Ok(false);};
            Ok(r.metadata.size_bytes <= crate::discovery::audio_byte_limit(self.max_audio_duration_secs)?
                && !r.segments.is_empty()
                && r.segments.iter().all(|s| matches!(s.locator, ResourceLocator::Time {start_ms,end_ms} if start_ms<end_ms && end_ms<=limit)))
        })
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

    /// Start a search request. Add more phrasings of the same need with
    /// [`SearchRequest::query`].
    pub fn search<'e>(&'e self, query: impl Into<String>) -> SearchRequest<'e> {
        SearchRequest {
            engine: self,
            queries: vec![query.into()],
            paths: Vec::new(),
            roots: Vec::new(),
            limit: 10,
            min_score: None,
            discovery: DiscoveryOptions::default(),
            preview_chars: Some(240),
            progress: None,
            sync: true,
            mode: SearchMode::default(),
            lexical_weight: None,
        }
    }

    /// Start an explicit indexing request (cache warming).
    pub fn index<'e>(&'e self) -> IndexRequest<'e> {
        IndexRequest {
            engine: self,
            paths: Vec::new(),
            roots: Vec::new(),
            discovery: DiscoveryOptions::default(),
            progress: None,
        }
    }

    /// Run a closure with exclusive access to the cache.
    /// The closure must not start indexing, which acquires the indexing lease
    /// before the cache mutex.
    pub fn with_cache<T>(&self, f: impl FnOnce(&mut dyn EmbeddingCache) -> Result<T>) -> Result<T> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| Error::Config("cache lock poisoned".into()))?;
        f(guard.as_mut())
    }

    /// Make sure every resource has up-to-date vectors in the cache.
    /// The caller keeps the indexing lease through any subsequent deletions.
    fn sync(
        &self,
        _indexing: &MutexGuard<'_, ()>,
        resources: &[ProvidedResource],
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
            if !self.supports_kind(r.kind) {
                stats.skipped.push(skip(
                    &r.id,
                    if r.kind == ResourceKind::Pdf && !cfg!(feature = "pdf") {
                        "PDF support is not compiled in (feature `pdf`)".into()
                    } else if r.kind == ResourceKind::Audio && !cfg!(feature = "audio") {
                        "audio support is not compiled in (feature `audio`)".into()
                    } else {
                        format!("{modality} search is not supported by this embedder")
                    },
                ));
                continue;
            }
            let descriptor_hash = if r.kind == ResourceKind::Pdf {
                #[cfg(feature = "pdf")]
                {
                    input_hash(&[
                        &r.fingerprint(),
                        crate::pdf::TEXT_VERSION,
                        crate::pdf::PREPARATION_VERSION,
                        &self.pdf_capabilities(),
                    ])
                }
                #[cfg(not(feature = "pdf"))]
                {
                    r.fingerprint()
                }
            } else if r.kind == ResourceKind::Audio {
                self.audio_descriptor_hash(r)
            } else {
                r.fingerprint()
            };
            let cached = known.get(&r.id);
            // Listing already supplies a freshness token; only stat again before a read.
            if cached.is_some_and(|m| {
                m.version.matches(&r.version)
                    && m.descriptor_hash.as_ref() == Some(&descriptor_hash)
            }) {
                if r.kind == ResourceKind::Audio && !self.audio_eligible(&r.id)? {
                    stats.skipped.push(skip(
                        &r.id,
                        "audio exceeds configured duration limit or is not validated",
                    ));
                } else {
                    stats.resources_unchanged += 1;
                }
                continue;
            }
            let version = match self.provider.version(&r.id) {
                Ok(v) => v,
                Err(e) => {
                    stats.skipped.push(skip(&r.id, e.to_string()));
                    continue;
                }
            };
            // A list/stat race must not attach old metadata to newly read bytes.
            if r.version.is_known() && !r.version.matches(&version) {
                stats
                    .skipped
                    .push(skip(&r.id, "resource changed after discovery"));
                continue;
            }
            #[cfg(feature = "audio")]
            if r.kind == ResourceKind::Audio {
                let source = self.provider.open(&r.id).and_then(|source| {
                    crate::audio::VerifiedReader::new(
                        source,
                        &r.id,
                        crate::discovery::audio_byte_limit(self.max_audio_duration_secs)?,
                    )
                });
                let mut source = match source {
                    Ok(s) => s,
                    Err(e) => {
                        stats.skipped.push(skip(&r.id, e.to_string()));
                        continue;
                    }
                };
                let after = match self.provider.version(&r.id) {
                    Ok(v) => v,
                    Err(e) => {
                        stats.skipped.push(skip(&r.id, e.to_string()));
                        continue;
                    }
                };
                if version.is_known() && !version.matches(&after) {
                    stats
                        .skipped
                        .push(skip(&r.id, "resource changed while reading"));
                    continue;
                }
                let content_hash = source.content_hash().to_owned();
                if let Some(m) = cached.filter(|m| {
                    m.metadata.content_hash == content_hash
                        && m.descriptor_hash.as_ref() == Some(&descriptor_hash)
                }) {
                    if !self.audio_eligible(&r.id)? {
                        stats.skipped.push(skip(
                            &r.id,
                            "audio exceeds configured duration limit or is not validated",
                        ));
                        continue;
                    }
                    let mut metadata = m.metadata.clone();
                    metadata.size_bytes = source.len();
                    metadata.modified_ns = r.modified_ns;
                    self.with_cache(|c| c.touch_resource(pid, &r.id, &metadata, &version))?;
                    stats.resources_unchanged += 1;
                    continue;
                }
                use std::io::{Read, Seek};
                let mut header = [0; 12];
                let header_result = source.read(&mut header).and_then(|n| {
                    source.rewind()?;
                    Ok(n)
                });
                let n = match header_result {
                    Ok(n) => n,
                    Err(_) => {
                        stats
                            .skipped
                            .push(skip(&r.id, "audio changed while reading"));
                        continue;
                    }
                };
                let target = Pending {
                    resource: r.id.clone(),
                    metadata: ResourceMetadata {
                        media_type: Some(crate::audio::media_type(&header[..n]).into()),
                        modality,
                        size_bytes: source.len(),
                        modified_ns: r.modified_ns,
                        content_hash,
                    },
                    version,
                    descriptor_hash,
                    jobs: Vec::new(),
                    page_diagnostics: Vec::new(),
                };
                if let Some(reason) = self.sync_audio(r, source, target, &mut stats, progress)? {
                    stats.skipped.push(skip(&r.id, reason));
                }
                continue;
            }
            let bytes = match self.provider.read(&r.id) {
                Ok(b) => b,
                Err(e) => {
                    stats.skipped.push(skip(&r.id, e.to_string()));
                    continue;
                }
            };
            let after = match self.provider.version(&r.id) {
                Ok(v) => v,
                Err(e) => {
                    stats.skipped.push(skip(&r.id, e.to_string()));
                    continue;
                }
            };
            if version.is_known() && !version.matches(&after) {
                stats
                    .skipped
                    .push(skip(&r.id, "resource changed while reading"));
                continue;
            }
            let content_hash = blake3::hash(&bytes).to_hex().to_string();
            if let Some(m) = cached.filter(|m| {
                m.metadata.content_hash == content_hash
                    && m.descriptor_hash.as_ref() == Some(&descriptor_hash)
            }) {
                let mut metadata = m.metadata.clone();
                metadata.size_bytes = bytes.len() as u64;
                metadata.modified_ns = r.modified_ns;
                self.with_cache(|c| c.touch_resource(pid, &r.id, &metadata, &version))?;
                stats.resources_unchanged += 1;
                continue;
            }
            let metadata = ResourceMetadata {
                media_type: match r.kind {
                    ResourceKind::Pdf => Some("application/pdf".into()),
                    #[cfg(feature = "audio")]
                    ResourceKind::Audio => Some(crate::audio::media_type(&bytes).into()),
                    _ => None,
                },
                modality,
                size_bytes: bytes.len() as u64,
                modified_ns: r.modified_ns,
                content_hash,
            };
            #[cfg(feature = "pdf")]
            if r.kind == ResourceKind::Pdf {
                let target = Pending {
                    resource: r.id.clone(),
                    metadata,
                    version,
                    descriptor_hash,
                    jobs: Vec::new(),
                    page_diagnostics: Vec::new(),
                };
                if let Some(reason) = self.sync_pdf(r, bytes, target, &mut stats, progress)? {
                    stats.skipped.push(skip(&r.id, reason));
                }
                continue;
            }
            let prepared = self.prepare(r, bytes);
            match prepared {
                Ok((jobs, page_diagnostics)) => pending.push(Pending {
                    resource: r.id.clone(),
                    metadata,
                    version,
                    descriptor_hash,
                    jobs,
                    page_diagnostics,
                }),
                Err(reason) => stats.skipped.push(skip(&r.id, reason)),
            }
        }

        // Only inputs the cache has never seen go to the model.
        let all_hashes: Vec<String> = pending
            .iter()
            .flat_map(|p| p.jobs.iter().map(|j| j.segment.input_hash.clone()))
            .collect();
        let already = self.with_cache(|c| c.known_inputs(pid, &all_hashes))?;
        let mut queued: HashSet<&str> = HashSet::new();
        let mut work: Vec<(usize, &Job)> = Vec::new();
        for (ri, p) in pending.iter().enumerate() {
            for j in &p.jobs {
                if !already.contains(&j.segment.input_hash)
                    && queued.insert(j.segment.input_hash.as_str())
                {
                    work.push((ri, j));
                }
            }
        }

        let total = work.len();
        let started = Instant::now();
        // Keyed by input, not resource: a deduplicated input belongs to every
        // resource that contains it, and all of them are incomplete if it fails.
        let mut failed: HashMap<String, String> = HashMap::new();
        let mut done = 0;
        let mut written = 0;
        if let Some(p) = progress {
            p(ProgressEvent::Embedding { done: 0, total });
        }
        for batch in work.chunks(self.batch_size) {
            let results = self.embed_batch(batch);
            let mut store = Vec::with_capacity(batch.len());
            for ((_, job), res) in batch.iter().zip(results) {
                match res {
                    Ok(v) => store.push((job.segment.input_hash.clone(), v)),
                    Err(e) => {
                        failed
                            .entry(job.segment.input_hash.clone())
                            .or_insert_with(|| e.to_string());
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
        stats.embed_ms += started.elapsed().as_millis() as u64;
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
        failed: &HashMap<String, String>,
        stats: &mut SyncStats,
    ) -> Result<usize> {
        let pid = self.profile_id.as_str();
        for p in pending.iter().take(upto).skip(from) {
            if let Some(reason) = p
                .jobs
                .iter()
                .find_map(|j| failed.get(&j.segment.input_hash))
            {
                stats.skipped.push(skip(&p.resource, reason.clone()));
                continue;
            }
            let resource = ResourceWrite {
                metadata: &p.metadata,
                version: &p.version,
                descriptor_hash: Some(&p.descriptor_hash),
                segments: p
                    .jobs
                    .iter()
                    .map(|j| SegmentWrite {
                        segment: &j.segment,
                        lexical_body: &j.lexical_body,
                    })
                    .collect(),
            };
            self.with_cache(|c| c.put_resource(pid, &p.resource, &resource))?;
            stats.resources_updated += 1;
            stats.page_diagnostics.extend(p.page_diagnostics.clone());
        }
        Ok(upto.max(from))
    }

    /// Validate backend output before associating vectors with requests or caching them.
    /// Preserve the backend's order and exact vector bytes without normalization.
    fn checked_embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        let vectors = self.embedder.embed_batch(requests)?;
        if vectors.len() != requests.len() {
            return Err(Error::Embedding(
                "embedder returned the wrong number of vectors".into(),
            ));
        }
        let dimension = self.embedder.dimension().get();
        if let Some(vector) = vectors.iter().find(|v| v.dim() != dimension) {
            return Err(Error::DimensionMismatch {
                expected: dimension,
                actual: vector.dim(),
            });
        }
        Ok(vectors)
    }

    /// Embed a batch; on failure retry one by one so a single bad input only skips its resource.
    fn embed_batch(&self, batch: &[(usize, &Job)]) -> Vec<Result<Vec<f32>>> {
        let inputs: Vec<EmbeddingRequest> = batch
            .iter()
            .map(|(_, j)| EmbeddingRequest {
                input: j.input.as_input(),
                task: EmbeddingTask::SearchDocument,
            })
            .collect();
        match self.checked_embed_batch(&inputs) {
            Ok(vs) => vs.into_iter().map(|v| Ok(v.into_vec())).collect(),
            Err(_) if inputs.len() > 1 => inputs
                .iter()
                .map(|i| {
                    self.checked_embed_batch(std::slice::from_ref(i))
                        .map(|mut v| v.remove(0).into_vec())
                })
                .collect(),
            Err(e) => vec![Err(e)],
        }
    }

    fn supports_kind(&self, kind: ResourceKind) -> bool {
        match kind {
            ResourceKind::Pdf => cfg!(feature = "pdf"),
            ResourceKind::Audio => {
                cfg!(feature = "audio") && self.embedder.supports(crate::Modality::Audio)
            }
            _ => self.embedder.supports(kind.modality()),
        }
    }

    fn supports_locator(&self, metadata: &ResourceMetadata, locator: &ResourceLocator) -> bool {
        if metadata.modality == crate::Modality::Audio {
            return cfg!(feature = "audio") && self.embedder.supports(crate::Modality::Audio);
        }
        if metadata.media_type.as_deref() != Some("application/pdf") {
            return true;
        }
        match locator {
            ResourceLocator::Page { .. } => {
                cfg!(feature = "pdf-render") && self.embedder.supports(crate::Modality::Image)
            }
            ResourceLocator::PageText { .. } => {
                cfg!(feature = "pdf") && self.embedder.supports(crate::Modality::Text)
            }
            _ => false,
        }
    }

    #[cfg(feature = "pdf")]
    fn pdf_capabilities(&self) -> String {
        #[cfg(feature = "pdf-render")]
        let renderer = crate::pdf::RENDER_VERSION;
        #[cfg(not(feature = "pdf-render"))]
        let renderer = "no-render";
        format!(
            "pdf-cap-v1;text={};image={};{renderer}",
            self.embedder.supports(crate::Modality::Text),
            self.pdf_images()
        )
    }

    #[cfg(feature = "pdf")]
    fn pdf_images(&self) -> bool {
        cfg!(feature = "pdf-render") && self.embedder.supports(crate::Modality::Image)
    }

    #[cfg(feature = "pdf")]
    fn sync_pdf(
        &self,
        r: &ProvidedResource,
        bytes: Vec<u8>,
        mut target: Pending,
        stats: &mut SyncStats,
        progress: Option<ProgressFn>,
    ) -> Result<Option<String>> {
        let pages = match crate::pdf::read_pages(&bytes, &r.id) {
            Ok(pages) => pages,
            Err(error) => return Ok(Some(pdf_reason(error))),
        };
        #[cfg(feature = "pdf-render")]
        let mut renderer = None;
        let capacity = self.batch_size.min(4);
        let mut batch = Vec::with_capacity(capacity);
        let mut records = Vec::new();
        for page in pages {
            if crate::pdf::usable(&page.text) {
                if !self.embedder.supports(crate::Modality::Text) {
                    target.page_diagnostics.push(PageDiagnostic {
                        resource: r.id.clone(),
                        page: page.page,
                        reason: "embedded text is not supported by this embedder".into(),
                    });
                    continue;
                }
                let digest = blake3::hash(page.text.as_bytes()).to_hex().to_string();
                for segment in self.segmenter.segment(&page.text, TextFormat::Plain) {
                    batch.push(Job::text(
                        ResourceLocator::PageText {
                            page: page.page,
                            byte_start: segment.byte_range.start as u64,
                            byte_end: segment.byte_range.end as u64,
                            text_version: crate::pdf::TEXT_VERSION.into(),
                            page_text_hash: digest.clone(),
                        },
                        None,
                        segment.text.clone(),
                        segment_title(r.title.as_deref(), None),
                        segment.text,
                    ));
                    if batch.len() == capacity
                        && let Some(reason) =
                            self.flush_media_batch(&mut batch, &mut records, stats, progress)?
                    {
                        return Ok(Some(reason));
                    }
                }
            } else if self.pdf_images() {
                #[cfg(feature = "pdf-render")]
                {
                    if renderer.is_none() {
                        renderer = match crate::pdf::render::PageRenderer::new(&bytes, &r.id) {
                            Ok(renderer) => Some(renderer),
                            Err(error) => return Ok(Some(pdf_reason(error))),
                        };
                    }
                    match renderer
                        .as_ref()
                        .expect("renderer initialized")
                        .page(page.page)
                    {
                        Ok(png) => {
                            batch.push(Job::image(ResourceLocator::Page { page: page.page }, png))
                        }
                        Err(Error::PdfRender { reason, .. }) => {
                            target.page_diagnostics.push(PageDiagnostic {
                                resource: r.id.clone(),
                                page: page.page,
                                reason: reason.into(),
                            });
                            continue;
                        }
                        Err(error) => return Ok(Some(pdf_reason(error))),
                    }
                }
            } else {
                target.page_diagnostics.push(PageDiagnostic {
                    resource: r.id.clone(), page: page.page,
                    reason: if cfg!(feature = "pdf-render") {
                        "no usable embedded text; image embedding is disabled"
                    } else {
                        "no usable embedded text; PDF rendering is not compiled in (feature `pdf-render`)"
                    }.into(),
                });
            }
            if batch.len() == capacity
                && let Some(reason) =
                    self.flush_media_batch(&mut batch, &mut records, stats, progress)?
            {
                return Ok(Some(reason));
            }
        }
        if let Some(reason) = self.flush_media_batch(&mut batch, &mut records, stats, progress)? {
            return Ok(Some(reason));
        }
        self.commit_media_resource(&target, &records, stats)?;
        stats.page_diagnostics.extend(target.page_diagnostics);
        Ok(None)
    }

    /// Publish only after preparation and the final embedding flush succeed.
    /// Earlier cached vectors may survive failure, but the prior binding must not change.
    #[cfg(any(feature = "pdf", feature = "audio"))]
    fn commit_media_resource(
        &self,
        target: &Pending,
        records: &[(CachedSegment, String)],
        stats: &mut SyncStats,
    ) -> Result<()> {
        let write = ResourceWrite {
            metadata: &target.metadata,
            version: &target.version,
            descriptor_hash: Some(&target.descriptor_hash),
            segments: records
                .iter()
                .map(|(segment, body)| SegmentWrite {
                    segment,
                    lexical_body: body,
                })
                .collect(),
        };
        self.with_cache(|c| c.put_resource(&self.profile_id, &target.resource, &write))?;
        stats.resources_updated += 1;
        Ok(())
    }

    #[cfg(any(feature = "pdf", feature = "audio"))]
    fn flush_media_batch(
        &self,
        batch: &mut Vec<Job>,
        records: &mut Vec<(CachedSegment, String)>,
        stats: &mut SyncStats,
        progress: Option<ProgressFn>,
    ) -> Result<Option<String>> {
        if batch.is_empty() {
            return Ok(None);
        }
        let hashes: Vec<_> = batch.iter().map(|j| j.segment.input_hash.clone()).collect();
        let already = self.with_cache(|c| c.known_inputs(&self.profile_id, &hashes))?;
        let mut queued = HashSet::new();
        let work: Vec<_> = batch
            .iter()
            .filter(|j| {
                !already.contains(&j.segment.input_hash) && queued.insert(&j.segment.input_hash)
            })
            .map(|j| (0, j))
            .collect();
        let mut store = Vec::new();
        let mut failure = None;
        if !work.is_empty() {
            if let Some(p) = progress {
                p(ProgressEvent::Embedding {
                    done: 0,
                    total: work.len(),
                });
            }
            let started = Instant::now();
            for ((_, job), result) in work.iter().zip(self.embed_batch(&work)) {
                match result {
                    Ok(vector) => store.push((job.segment.input_hash.clone(), vector)),
                    Err(error) => {
                        failure.get_or_insert_with(|| error.to_string());
                    }
                }
            }
            stats.embed_ms += started.elapsed().as_millis() as u64;
            self.with_cache(|c| c.put_embeddings(&self.profile_id, &store))?;
            stats.segments_embedded += store.len();
            if let Some(p) = progress {
                p(ProgressEvent::Embedding {
                    done: work.len(),
                    total: work.len(),
                });
            }
        }
        // Retain only compact locators and transient lexical text. Drop every
        // PNG and model input here before the next bounded batch is rendered.
        records.extend(batch.drain(..).map(|j| (j.segment, j.lexical_body)));
        Ok(failure)
    }

    #[cfg(feature = "audio")]
    fn sync_audio(
        &self,
        r: &ProvidedResource,
        source: crate::audio::VerifiedReader,
        target: Pending,
        stats: &mut SyncStats,
        progress: Option<ProgressFn>,
    ) -> Result<Option<String>> {
        let mut batch = Vec::new();
        let mut records = Vec::new();
        let capacity = self.batch_size.min(4);
        let result = crate::audio::windows_verified(
            source,
            &r.id,
            self.max_audio_duration_secs,
            |locator, samples| {
                let mut hash = blake3::Hasher::new();
                hash.update(b"audio-input-v1");
                hash.update(crate::audio::PREPARATION.as_bytes());
                let identity = self.embedder.input_identity(crate::Modality::Audio);
                hash.update(&(identity.len() as u64).to_le_bytes());
                hash.update(identity.as_bytes());
                hash.update(&(samples.len() as u64).to_le_bytes());
                for sample in &samples {
                    hash.update(&sample.to_le_bytes());
                }
                batch.push(Job {
                    segment: CachedSegment {
                        locator,
                        heading: None,
                        input_hash: hash.finalize().to_hex().to_string(),
                    },
                    lexical_body: String::new(),
                    input: JobInput::Audio { samples },
                });
                if batch.len() == capacity
                    && let Some(reason) =
                        self.flush_media_batch(&mut batch, &mut records, stats, progress)?
                {
                    return Err(Error::Embedding(reason));
                }
                Ok(())
            },
        );
        if let Err(error) = result {
            return match error {
                Error::ChangedWhileReading { .. }
                | Error::UnsupportedAudio { .. }
                | Error::InvalidLocator { .. }
                | Error::Embedding(_)
                | Error::ModelNotFound(_) => Ok(Some(error.to_string())),
                other => Err(other),
            };
        }
        if let Some(reason) = self.flush_media_batch(&mut batch, &mut records, stats, progress)? {
            return Ok(Some(reason));
        }
        let after = match self.provider.version(&r.id) {
            Ok(v) => v,
            Err(e) => return Ok(Some(e.to_string())),
        };
        if target.version.is_known() && !target.version.matches(&after) {
            return Ok(Some("resource changed while reading".into()));
        }
        self.commit_media_resource(&target, &records, stats)?;
        Ok(None)
    }

    /// Turn a resource into embedding jobs. Errors are reasons to skip the resource.
    fn prepare(
        &self,
        r: &ProvidedResource,
        bytes: Vec<u8>,
    ) -> std::result::Result<(Vec<Job>, Vec<PageDiagnostic>), String> {
        match r.kind {
            ResourceKind::Audio => {
                Err("audio requires the streaming preparation path (feature `audio`)".into())
            }
            ResourceKind::Pdf => {
                #[cfg(feature = "pdf")]
                {
                    Err("PDF resources require the streaming preparation path".into())
                }
                #[cfg(not(feature = "pdf"))]
                {
                    Err("PDF feature is disabled".into())
                }
            }
            ResourceKind::Text(format) => {
                if bytes.iter().take(8192).any(|&b| b == 0) {
                    return Err("looks like a binary file".into());
                }
                let text = String::from_utf8(bytes).map_err(|_| "not valid UTF-8".to_string())?;
                let stem = &r.title;
                Ok((
                    self.segmenter
                        .segment(&text, format)
                        .into_iter()
                        .map(|s| {
                            let title = segment_title(stem.as_deref(), s.heading.as_deref());
                            let model_text = match format {
                                TextFormat::Markdown => strip_link_targets(&s.text),
                                TextFormat::Plain | TextFormat::Code => s.text.clone(),
                            };
                            Job::text(
                                ResourceLocator::Text {
                                    byte_start: s.byte_range.start as u64,
                                    byte_end: s.byte_range.end as u64,
                                    line_start: Some(s.line_start),
                                    line_end: Some(s.line_end),
                                },
                                s.heading,
                                // Index the original source, including link targets.
                                s.text,
                                title,
                                model_text,
                            )
                        })
                        .collect(),
                    Vec::new(),
                ))
            }
            ResourceKind::Image => Ok((
                vec![Job::image(ResourceLocator::WholeResource, bytes)],
                Vec::new(),
            )),
        }
    }
}

#[cfg(feature = "pdf")]
fn pdf_reason(error: Error) -> String {
    match error {
        Error::PdfParse { reason, .. } => format!("PDF parse failed: {reason}"),
        Error::PdfRender { reason, .. } => format!("PDF rendering failed: {reason}"),
        Error::ParserPanicked { .. } => "PDF parser panicked".into(),
        other => other.to_string(),
    }
}

struct Pending {
    descriptor_hash: String,
    resource: ResourceId,
    version: ResourceVersion,
    metadata: ResourceMetadata,
    jobs: Vec<Job>,
    page_diagnostics: Vec<PageDiagnostic>,
}

struct Job {
    /// `segment.input_hash` identifies the model input; identical inputs in
    /// different resources are embedded once.
    segment: CachedSegment,
    lexical_body: String,
    input: JobInput,
}

impl Job {
    fn image(locator: ResourceLocator, bytes: Vec<u8>) -> Self {
        Self {
            segment: CachedSegment {
                locator,
                heading: None,
                input_hash: input_hash(&["image", &blake3::hash(&bytes).to_hex()]),
            },
            lexical_body: String::new(),
            input: JobInput::Image { bytes },
        }
    }

    /// Derive cache identity from exactly the title and text sent to the model.
    /// Lexical text may retain source syntax that model preparation removes.
    fn text(
        locator: ResourceLocator,
        heading: Option<String>,
        lexical_body: String,
        title: Option<String>,
        text: String,
    ) -> Self {
        Self {
            segment: CachedSegment {
                locator,
                heading,
                input_hash: input_hash(&["text", title.as_deref().unwrap_or(""), &text]),
            },
            lexical_body,
            input: JobInput::Text { title, text },
        }
    }
}

enum JobInput {
    Text {
        title: Option<String>,
        text: String,
    },
    Image {
        bytes: Vec<u8>,
    },
    #[cfg(feature = "audio")]
    Audio {
        samples: Vec<f32>,
    },
}

impl JobInput {
    fn as_input(&self) -> ResourceInput<'_> {
        match self {
            JobInput::Text { title, text } => ResourceInput::Text {
                title: title.as_deref(),
                text,
            },
            JobInput::Image { bytes } => ResourceInput::Image { bytes },
            #[cfg(feature = "audio")]
            JobInput::Audio { samples } => ResourceInput::Audio {
                samples,
                sample_rate: 16000,
            },
        }
    }
}

/// Replace Markdown links with their visible text before embedding.
///
/// `[dossier](../people/Ada.md)` becomes `dossier`, and
/// `[[Ada - old|old notes]]` becomes `old notes`. Link targets are paths and
/// URLs, not prose; left in, they made link-heavy lines match unrelated queries.
/// Only the model input changes: locators still point at the original bytes.
fn strip_link_targets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('[') {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        if let Some(inner) = after.strip_prefix("[[")
            && let Some(end) = inner.find("]]")
            && !inner[..end].contains('\n')
        {
            let target = &inner[..end];
            out.push_str(target.rsplit('|').next().unwrap_or(target));
            rest = &inner[end + 2..];
            continue;
        }
        if let Some(close) = after.find("](")
            && !after[..close].contains('\n')
            && let Some(end) = after[close + 2..].find(')')
            && !after[close + 2..close + 2 + end].contains(char::is_whitespace)
        {
            out.push_str(&after[1..close]);
            rest = &after[close + 2 + end + 1..];
            continue;
        }
        out.push('[');
        rest = &after[1..];
    }
    out.push_str(rest);
    out
}

fn skip(resource: &ResourceId, reason: impl Into<String>) -> SkippedResource {
    SkippedResource {
        resource: resource.clone(),
        path: resource.to_file_path(),
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
    queries: Vec<String>,
    paths: Vec<PathBuf>,
    roots: Vec<ResourceId>,
    limit: usize,
    min_score: Option<f32>,
    discovery: DiscoveryOptions,
    preview_chars: Option<usize>,
    progress: Option<ProgressFn<'e>>,
    sync: bool,
    mode: SearchMode,
    lexical_weight: Option<f32>,
}

impl<'e> SearchRequest<'e> {
    /// Add provider-specific roots without filesystem path conversion.
    pub fn roots(mut self, roots: impl IntoIterator<Item = ResourceId>) -> Self {
        self.roots.extend(roots);
        self
    }

    /// Search for another phrasing too. Each segment is ranked by its best score
    /// over all queries, so rephrasing widens recall without diluting scores.
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.queries.push(query.into());
        self
    }

    /// Add filesystem-only paths, canonicalized locally at execution even with
    /// a custom provider. Use `roots()` for generic provider identities.
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

    /// Override how much keyword rank counts against semantic rank in hybrid
    /// mode (semantic is 1). By default it depends on the query, see
    /// [`keyword_weight`]. Must be finite and non-negative; zero disables the
    /// keyword signal in hybrid mode. Valid overrides are ignored in other modes.
    pub fn lexical_weight(mut self, w: f32) -> Self {
        self.lexical_weight = Some(w);
        self
    }

    pub fn mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }

    /// Hide semantic matches below this cosine similarity. Keyword and file
    /// name matches are kept: their relevance is not a cosine.
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

    /// With `false`, search only what is already cached and embed nothing. For
    /// callers where a separate indexer (see `watch`) keeps the cache fresh, so a
    /// search never waits for a large first index. Changed files may be stale.
    pub fn sync(mut self, sync: bool) -> Self {
        self.sync = sync;
        self
    }

    pub fn execute(self) -> Result<SearchResults> {
        let started = Instant::now();
        let engine = self.engine;
        if self
            .lexical_weight
            .is_some_and(|w| !w.is_finite() || w < 0.0)
        {
            return Err(Error::Config(
                "lexical weight must be finite and non-negative".into(),
            ));
        }
        if self.paths.is_empty() && self.roots.is_empty() {
            return Err(Error::Config("no paths to search".into()));
        }
        let mut discovery = self.discovery;
        discovery.max_audio_duration_secs = engine.max_audio_duration_secs;
        let scope = ResourceScope::from_paths(&self.paths, self.roots, discovery)?;
        let resources = engine.provider.list(&scope)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let sync = if self.sync {
            let indexing = engine
                .indexing
                .lock()
                .map_err(|_| Error::Config("indexing lock poisoned".into()))?;
            engine.sync(&indexing, &resources, self.progress)?
        } else {
            SyncStats {
                resources_discovered: resources.len(),
                ..Default::default()
            }
        };

        let pid = engine.profile_id.as_str();
        let mut wanted: HashSet<ResourceId> = resources
            .iter()
            .filter(|r| engine.supports_kind(r.kind))
            .map(|r| r.id.clone())
            .collect();
        let metadata = engine.with_cache(|c| c.resource_index(pid))?;
        for r in resources.iter().filter(|r| r.kind == ResourceKind::Audio) {
            let failed = sync.skipped.iter().any(|s| s.resource == r.id);
            let descriptor = engine.audio_descriptor_hash(r);
            let current = metadata.get(&r.id).is_some_and(|m| {
                // Unknown/changed cached sources cannot establish eligibility in cached-only search.
                (self.sync || m.version.matches(&r.version))
                    && m.metadata.size_bytes == r.size_bytes
                    && m.descriptor_hash.as_ref() == Some(&descriptor)
            });
            if failed || !current || !engine.audio_eligible(&r.id)? {
                wanted.remove(&r.id);
            }
        }

        // Each signal contributes a deeper list than asked for, so a hit that is
        // only moderately good on two signals can still surface.
        let depth = (self.limit * 4).max(50);

        let mut segments = SegmentTable::default();
        let mut semantic: Vec<(SegmentKey, f32, usize)> = Vec::new();
        let mut searched = 0;
        if self.mode != SearchMode::Lexical {
            let dim = engine.embedder.dimension().get();
            let mut query_vectors = Vec::with_capacity(self.queries.len());
            for q in &self.queries {
                let v = engine.embedder.embed_text(q, EmbeddingTask::SearchQuery)?;
                if v.dim() != dim {
                    return Err(Error::DimensionMismatch {
                        expected: dim,
                        actual: v.dim(),
                    });
                }
                query_vectors.push(v);
            }
            let mut stored = engine.with_cache(|c| c.vectors_for(pid, &wanted))?;
            // Vector IDs break cosine ties, including at the top-k cutoff.
            stored.sort_by_cached_key(|v| (crate::cache::resource_key(&v.resource), v.ordinal));
            let mut index = BruteForceIndex::with_capacity(dim, stored.len())?;
            let mut keys = Vec::with_capacity(stored.len());
            for v in stored {
                // A wrong-sized vector means a corrupt cache; never compare it.
                if v.vector.len() == dim
                    && metadata
                        .get(&v.resource)
                        .is_some_and(|m| engine.supports_locator(&m.metadata, &v.segment.locator))
                {
                    index.add(&v.vector);
                    keys.push(segments.insert(v.resource, v.ordinal, v.segment));
                }
            }
            searched = keys.len();
            let min_score = self.min_score.unwrap_or(f32::MIN);
            let take = if self.mode == SearchMode::Semantic {
                self.limit
            } else {
                depth
            };
            semantic = fuse_max(
                query_vectors
                    .iter()
                    .map(|q| index.search(q.as_slice(), take, min_score)),
                take,
            )
            .into_iter()
            .map(|f| (keys[f.scored.id].clone(), f.scored.score, f.query_index))
            .collect();
        }

        let mut lexical: Vec<(SegmentKey, f32)> = Vec::new();
        let mut named: Vec<SegmentKey> = Vec::new();
        if self.mode != SearchMode::Semantic {
            // Semantic candidates already carry locators in vectors_for; only
            // lexical search needs this separate eligibility lookup.
            // Lexical caches limit their result list before the engine sees it.
            // Reserve room for every ineligible PDF segment, then filter before
            // restoring the candidate depth. A disabled scan cannot exhaust it.
            let mut unsupported = HashSet::new();
            // These are the two locator kinds emitted by PDF preparation. Both
            // feature gates AND model capabilities must hold, including when a
            // different model configuration reuses this profile with sync(false).
            let all_pdf_locators_supported = cfg!(feature = "pdf-render")
                && engine.embedder.supports(crate::Modality::Image)
                && cfg!(feature = "pdf")
                && engine.embedder.supports(crate::Modality::Text);
            if !all_pdf_locators_supported {
                for resource in &wanted {
                    if let Some(state) = metadata.get(resource)
                        && state.metadata.media_type.as_deref() == Some("application/pdf")
                    {
                        for &ordinal in &segments.load_resource(engine, pid, resource)?.unsupported
                        {
                            unsupported.insert((resource.clone(), ordinal));
                        }
                    }
                }
            }
            let query = LexicalQuery::new(self.queries.iter().map(String::as_str));
            for hit in engine.with_cache(|c| {
                c.lexical_search(
                    pid,
                    &query,
                    &wanted,
                    depth.saturating_add(unsupported.len()),
                )
            })? {
                let key = (hit.resource, hit.ordinal);
                if unsupported.contains(&key) {
                    continue;
                }
                if metadata
                    .get(&key.0)
                    .is_some_and(|m| m.metadata.media_type.as_deref() == Some("application/pdf"))
                {
                    let Some(segment) = segments.get_or_load(engine, pid, &key)? else {
                        continue;
                    };
                    if !engine.supports_locator(&metadata[&key.0].metadata, &segment.locator) {
                        continue;
                    }
                }
                lexical.push((key, hit.score));
                if lexical.len() == depth {
                    break;
                }
            }
            for resource in resources.iter().filter(|r| {
                wanted.contains(&r.id)
                    && self
                        .queries
                        .iter()
                        .any(|q| names_file(q, Path::new(&r.id.labels().1)))
            }) {
                if !metadata.contains_key(&resource.id) {
                    continue;
                }
                // A filename identifies the first supported cached location,
                // which need not be ordinal zero in a mixed PDF.
                if let Some(ordinal) = segments
                    .load_resource(engine, pid, &resource.id)?
                    .first_supported
                {
                    named.push((resource.id.clone(), ordinal));
                }
            }
            if self.mode == SearchMode::Lexical {
                searched = lexical.len();
            }
        }

        let weight = match self.mode {
            SearchMode::Hybrid => self
                .lexical_weight
                .unwrap_or_else(|| keyword_weight(&self.queries)),
            _ => keyword_weight(&self.queries),
        };
        let ranked = rank(self.mode, &semantic, &lexical, weight, &named, self.limit);

        let mut previews = PreviewReader::default();
        let mut hits = Vec::with_capacity(ranked.len());
        for (key, score, signals, query_index) in ranked {
            let Some(metadata) = metadata.get(&key.0).map(|s| s.metadata.clone()) else {
                continue;
            };
            let Some(segment) = segments.get_or_load(engine, pid, &key)? else {
                continue;
            };
            if !engine.supports_locator(&metadata, &segment.locator) {
                continue;
            }
            let preview = self.preview_chars.and_then(|n| {
                previews.preview(
                    engine.provider.as_ref(),
                    &key.0,
                    &segment.locator,
                    &metadata,
                    n,
                )
            });
            hits.push(SearchHit {
                resource: key.0,
                locator: segment.locator.clone(),
                score,
                signals,
                metadata,
                heading: segment.heading.clone(),
                preview,
                query_index,
            });
        }

        Ok(SearchResults {
            queries: self.queries,
            mode: self.mode,
            hits,
            stats: SearchStats {
                sync,
                segments_searched: searched,
                total_ms: started.elapsed().as_millis() as u64,
            },
        })
    }
}

/// A segment's identity within a search: its resource and position.
type SegmentKey = (ResourceId, usize);

/// Segments seen so far, filled from the vectors already loaded and, for hits
/// found only by keyword or name, from the cache on demand.
#[derive(Default)]
struct SegmentTable {
    segments: HashMap<ResourceId, HashMap<usize, CachedSegment>>,
    // Vector rows are only a partial view: keyword-only ordinals may be absent.
    complete: HashMap<ResourceId, CompleteResource>,
}

/// Eligibility from the complete lookup, independent of earlier vector evidence.
/// A missing resource has no filename candidate, even if scored rows survive.
#[derive(Default)]
struct CompleteResource {
    unsupported: Vec<usize>,
    first_supported: Option<usize>,
}

impl SegmentTable {
    fn insert(&mut self, path: ResourceId, ordinal: usize, segment: CachedSegment) -> SegmentKey {
        self.segments
            .entry(path.clone())
            .or_default()
            .insert(ordinal, segment);
        (path, ordinal)
    }

    /// Bulk-ingest one complete lookup, including a missing/empty resource.
    /// Errors leave completion unset so a later attempt can retry.
    fn load_resource(
        &mut self,
        engine: &SearchEngine,
        pid: &str,
        resource: &ResourceId,
    ) -> Result<&CompleteResource> {
        if !self.complete.contains_key(resource) {
            let cached = engine.with_cache(|c| c.resource(pid, resource))?;
            let mut complete = CompleteResource::default();
            let seen = self.segments.entry(resource.clone()).or_default();
            if let Some(cached) = cached {
                for (ordinal, segment) in cached.segments.into_iter().enumerate() {
                    if engine.supports_locator(&cached.metadata, &segment.locator) {
                        complete.first_supported.get_or_insert(ordinal);
                    } else {
                        complete.unsupported.push(ordinal);
                    }
                    // A concurrent indexer can replace/remove a resource after
                    // vectors_for. Never relabel or discard already-scored evidence.
                    seen.entry(ordinal).or_insert(segment);
                }
            }
            self.complete.insert(resource.clone(), complete);
        }
        Ok(&self.complete[resource])
    }

    fn get_or_load(
        &mut self,
        engine: &SearchEngine,
        pid: &str,
        key: &SegmentKey,
    ) -> Result<Option<&CachedSegment>> {
        if !self
            .segments
            .get(&key.0)
            .is_some_and(|s| s.contains_key(&key.1))
        {
            self.load_resource(engine, pid, &key.0)?;
        }
        Ok(self.segments.get(&key.0).and_then(|s| s.get(&key.1)))
    }
}

/// Whether `query` names this file: its name (`father.md`), its stem (`father`)
/// or a trailing part of its path (`notes/father.md`), ignoring case.
fn names_file(query: &str, path: &Path) -> bool {
    let q = query
        .trim()
        .trim_matches(['"', '\'', '`'])
        .trim()
        .replace('\\', "/")
        .to_lowercase();
    if q.is_empty() {
        return false;
    }
    let full = path.to_string_lossy().replace('\\', "/").to_lowercase();
    let path = Path::new(&full);
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase());
    let stem = path.file_stem().map(|n| n.to_string_lossy().to_lowercase());
    if name.as_deref() == Some(q.as_str()) || stem.as_deref() == Some(q.as_str()) {
        return true;
    }
    q.contains('/') && {
        full == q || (full.ends_with(&q) && full[..full.len() - q.len()].ends_with('/'))
    }
}

/// How much keyword rank counts, relative to semantic rank, for these queries.
///
/// Names and identifiers (`father.md`, `INC-18372`, `Session::restore()`) or a
/// word or two are what keywords are for, so they count fully. In a question
/// in prose, shared words are mostly incidental ("old", "match"), so they only
/// break ties between hits of similar meaning.
pub fn keyword_weight(queries: &[String]) -> f32 {
    let identifier = |w: &str| {
        w.chars().any(|c| c.is_ascii_digit())
            || w.trim_matches(|c: char| !c.is_alphanumeric())
                .contains(['.', '_', ':', '-', '/', '#'])
    };
    let keywordish = queries.iter().any(|q| {
        let words: Vec<&str> = q.split_whitespace().collect();
        words.iter().any(|w| identifier(w)) || LexicalQuery::new([q.as_str()]).groups.len() <= 2
    });
    if keywordish {
        1.0
    } else {
        PROSE_KEYWORD_WEIGHT
    }
}

/// Keyword weight for questions in prose; see [`keyword_weight`].
pub const PROSE_KEYWORD_WEIGHT: f32 = 0.3;

/// Reciprocal rank fusion constant: damps the gap between the first few ranks
/// so one signal's top hit cannot drown out a hit two signals agree on.
const RRF_K: f32 = 60.0;
/// A file the query names outweighs a semantic and a keyword first place.
const FILENAME_WEIGHT: f32 = 3.0;

/// Merge the signals into the final ranking.
fn rank(
    mode: SearchMode,
    semantic: &[(SegmentKey, f32, usize)],
    lexical: &[(SegmentKey, f32)],
    lexical_weight: f32,
    named: &[SegmentKey],
    limit: usize,
) -> Vec<(SegmentKey, f32, MatchSignals, usize)> {
    if mode == SearchMode::Semantic {
        return semantic
            .iter()
            .take(limit)
            .map(|(k, cos, q)| {
                let signals = MatchSignals {
                    semantic: Some(*cos),
                    ..Default::default()
                };
                (k.clone(), *cos, signals, *q)
            })
            .collect();
    }
    let rrf = |rank: usize, weight: f32| weight / (RRF_K + rank as f32 + 1.0);
    let mut fused: HashMap<SegmentKey, (f32, MatchSignals, usize)> = HashMap::new();
    for (rank, (k, cos, q)) in semantic.iter().enumerate() {
        let e = fused.entry(k.clone()).or_default();
        e.0 += rrf(rank, 1.0);
        e.1.semantic = Some(*cos);
        e.2 = *q;
    }
    if lexical_weight > 0.0 {
        for (rank, (k, bm25)) in lexical.iter().enumerate() {
            let e = fused.entry(k.clone()).or_default();
            e.0 += rrf(rank, lexical_weight);
            e.1.lexical = Some(*bm25);
        }
    }
    for (rank, k) in named.iter().enumerate() {
        let e = fused.entry(k.clone()).or_default();
        e.0 += rrf(rank, FILENAME_WEIGHT);
        e.1.filename = true;
    }
    // The best possible score: first place on every signal that found anything.
    let best = [
        (!semantic.is_empty(), 1.0),
        (!lexical.is_empty(), lexical_weight),
        (!named.is_empty(), FILENAME_WEIGHT),
    ]
    .iter()
    .filter(|(found, w)| *found && *w > 0.0)
    .map(|(_, w)| rrf(0, *w))
    .sum::<f32>();
    if best == 0.0 {
        return Vec::new();
    }
    let mut out: Vec<_> = fused
        .into_iter()
        .map(|(k, (score, signals, q))| (k, score / best, signals, q))
        .collect();
    out.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| {
                b.2.semantic
                    .unwrap_or(f32::MIN)
                    .total_cmp(&a.2.semantic.unwrap_or(f32::MIN))
            })
            .then_with(|| a.0.cmp(&b.0))
    });
    out.truncate(limit);
    out
}

struct Fused {
    scored: Scored,
    query_index: usize,
}

/// Merge per-query top-k lists, keeping each segment's best score.
///
/// Exact for the fused top-k: a segment whose best score comes from query q is
/// beaten on q by fewer than `limit` segments, so it is already in q's top-k.
fn fuse_max(per_query: impl Iterator<Item = Vec<Scored>>, limit: usize) -> Vec<Fused> {
    let mut best: HashMap<usize, Fused> = HashMap::new();
    for (query_index, hits) in per_query.enumerate() {
        for scored in hits {
            match best.get(&scored.id) {
                Some(b) if b.scored.score >= scored.score => {}
                _ => {
                    best.insert(
                        scored.id,
                        Fused {
                            scored,
                            query_index,
                        },
                    );
                }
            }
        }
    }
    let mut fused: Vec<Fused> = best.into_values().collect();
    fused.sort_by(|a, b| {
        b.scored
            .score
            .total_cmp(&a.scored.score)
            .then(a.scored.id.cmp(&b.scored.id))
    });
    fused.truncate(limit);
    fused
}

pub struct IndexRequest<'e> {
    engine: &'e SearchEngine,
    paths: Vec<PathBuf>,
    roots: Vec<ResourceId>,
    discovery: DiscoveryOptions,
    progress: Option<ProgressFn<'e>>,
}

impl<'e> IndexRequest<'e> {
    /// Add provider-specific roots without filesystem path conversion.
    pub fn roots(mut self, roots: impl IntoIterator<Item = ResourceId>) -> Self {
        self.roots.extend(roots);
        self
    }

    /// Add filesystem-only paths, canonicalized locally at execution even with
    /// a custom provider. Use `roots()` for generic provider identities.
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
        if self.paths.is_empty() && self.roots.is_empty() {
            return Err(Error::Config("no paths to index".into()));
        }
        let mut discovery = self.discovery;
        discovery.max_audio_duration_secs = engine.max_audio_duration_secs;
        let scope = ResourceScope::from_paths(&self.paths, self.roots, discovery)?;
        let resources = engine.provider.list(&scope)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let indexing = engine
            .indexing
            .lock()
            .map_err(|_| Error::Config("indexing lock poisoned".into()))?;
        let mut sync = engine.sync(&indexing, &resources, self.progress)?;

        let pid = engine.profile_id.as_str();
        let present: HashSet<&ResourceId> = resources.iter().map(|r| &r.id).collect();
        let known = engine.with_cache(|c| c.resource_index(pid))?;
        let mut gone = Vec::new();
        match engine.provider.deletion_scope(&scope) {
            Ok(deletion_scope) => {
                for id in known.into_keys() {
                    if present.contains(&id) || !deletion_scope.contains(&id) {
                        continue;
                    }
                    match engine.provider.is_deleted(&id, &deletion_scope) {
                        Ok(true) => gone.push(id),
                        Ok(false) => {}
                        Err(e) => sync.skipped.push(skip(&id, format!("deletion check: {e}"))),
                    }
                }
            }
            Err(e) => {
                for root in &scope.roots {
                    sync.skipped
                        .push(skip(root, format!("deletion scope: {e}")));
                }
            }
        }
        let removed = engine.with_cache(|c| {
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
    /// Each file read once per search, with the digest of exactly those bytes.
    files: HashMap<ResourceId, Option<ResourceBytes>>,
    #[cfg(feature = "pdf")]
    pages: HashMap<(ResourceId, u32), Option<String>>,
    #[cfg(feature = "pdf")]
    page_bytes: usize,
    #[cfg(all(test, feature = "pdf"))]
    page_decodes: usize,
}

impl PreviewReader {
    fn preview(
        &mut self,
        provider: &dyn ResourceProvider,
        path: &ResourceId,
        locator: &ResourceLocator,
        meta: &ResourceMetadata,
        max_chars: usize,
    ) -> Option<String> {
        if !matches!(
            locator,
            ResourceLocator::Text { .. } | ResourceLocator::PageText { .. }
        ) {
            return None;
        }
        let read = self
            .files
            .entry(path.clone())
            .or_insert_with(|| ResourceBytes::read(provider, path).ok())
            .as_ref()?;
        let bytes = read.verify(path, meta).ok()?;
        let text = match locator {
            ResourceLocator::Text { .. } => text_range(bytes, path, locator).ok()?,
            #[cfg(feature = "pdf")]
            ResourceLocator::PageText { page, .. }
                if meta.media_type.as_deref() == Some("application/pdf") =>
            {
                let key = (path.clone(), *page);
                if !self.pages.contains_key(&key) {
                    #[cfg(test)]
                    {
                        self.page_decodes += 1;
                    }
                    let text = crate::pdf::read_page(bytes, path, *page).ok();
                    if let Some(t) = &text {
                        if self.page_bytes + t.len() > crate::pdf::MAX_DOCUMENT_TEXT_BYTES {
                            self.pages.insert(key, None);
                            return None;
                        }
                        self.page_bytes += t.len();
                    }
                    self.pages.insert(key.clone(), text);
                }
                crate::pdf::slice(self.pages.get(&key)?.as_deref()?, path, locator).ok()?
            }
            _ => return None,
        };
        Some(make_preview(text, max_chars))
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

    #[cfg(any(feature = "pdf", feature = "audio"))]
    #[test]
    fn media_commit_failure_preserves_binding_and_update_count() {
        struct NoInference(crate::ModelIdentity);
        impl Embedder for NoInference {
            fn identity(&self) -> &crate::ModelIdentity {
                &self.0
            }
            fn supports(&self, _: crate::Modality) -> bool {
                true
            }
            fn embed_batch(
                &self,
                _: &[EmbeddingRequest<'_>],
            ) -> seyk_embed::Result<Vec<Embedding>> {
                panic!("committing prepared records must not run inference")
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.db");
        let engine = SearchEngine::builder()
            .embedder(NoInference(crate::ModelIdentity {
                model_id: "test/media-commit".into(),
                revision: "1".into(),
                variant: "test".into(),
                dimension: crate::Dimension::D128,
                normalization: "l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            }))
            .cache(crate::SqliteCache::open(&path).unwrap())
            .build()
            .unwrap();
        let mut target = Pending {
            resource: "test://media/document".parse().unwrap(),
            metadata: ResourceMetadata {
                media_type: Some("application/pdf".into()),
                modality: crate::Modality::Document,
                size_bytes: 10,
                modified_ns: None,
                content_hash: "old-content".into(),
            },
            version: ResourceVersion::known("old-version"),
            descriptor_hash: "old-descriptor".into(),
            jobs: Vec::new(),
            page_diagnostics: Vec::new(),
        };
        let records = vec![(
            CachedSegment {
                locator: ResourceLocator::Page { page: 1 },
                heading: None,
                input_hash: "old-input".into(),
            },
            String::new(),
        )];
        let mut stats = SyncStats::default();
        engine
            .commit_media_resource(&target, &records, &mut stats)
            .unwrap();
        assert_eq!(stats.resources_updated, 1);
        let binding = || {
            engine
                .with_cache(|cache| cache.resource(engine.profile_id(), &target.resource))
                .unwrap()
                .unwrap()
        };
        let old = binding();
        target.metadata.content_hash = "new-content".into();
        target.version = ResourceVersion::known("new-version");
        target.descriptor_hash = "new-descriptor".into();
        let conn = rusqlite::Connection::open(&path).unwrap();
        // Fail after put_resource has deleted the old binding inside its transaction.
        conn.execute_batch(
            "CREATE TRIGGER fail_media_commit BEFORE INSERT ON resources
             BEGIN SELECT RAISE(ABORT, 'injected media commit failure'); END;",
        )
        .unwrap();
        let error = engine
            .commit_media_resource(&target, &[], &mut stats)
            .unwrap_err();
        assert!(matches!(error, Error::Cache(_)));
        assert!(error.to_string().contains("injected media commit failure"));
        assert_eq!(stats.resources_updated, 1);
        assert_eq!(binding(), old);

        conn.execute_batch("DROP TRIGGER fail_media_commit;")
            .unwrap();
        engine
            .commit_media_resource(&target, &[], &mut stats)
            .unwrap();
        assert_eq!(stats.resources_updated, 2);
        let new = binding();
        assert_eq!(new.metadata, target.metadata);
        assert_eq!(new.version, target.version);
        assert_eq!(new.descriptor_hash.as_deref(), Some("new-descriptor"));
        assert!(new.segments.is_empty());
    }

    #[test]
    fn keywords_count_fully_for_names_and_identifiers_only() {
        let w = |q: &str| keyword_weight(&[q.to_string()]);
        for q in [
            "father.md",
            "Seren",
            "INC-18372",
            "where is Session::restore() called",
            "invoice_38192",
            "pricing page",
        ] {
            assert_eq!(w(q), 1.0, "{q}");
        }
        for q in [
            "where did I write about someone becoming like their father",
            "grief as something physical",
        ] {
            assert_eq!(w(q), PROSE_KEYWORD_WEIGHT, "{q}");
        }
    }

    #[test]
    fn names_file_matches_name_stem_and_path_suffix_only() {
        let p = Path::new("/book/people/father.md");
        for q in ["father.md", "Father", "people/father.md", " `father.md` "] {
            assert!(names_file(q, p), "{q}");
        }
        for q in ["ather.md", "eople/father.md", "my father", "father.txt", ""] {
            assert!(!names_file(q, p), "{q}");
        }
    }

    #[test]
    fn names_file_normalizes_both_paths_and_queries_and_preserves_boundaries() {
        for path in [
            "/book/my people/My Notes.md",
            r"C:\book\my people\My Notes.md",
        ] {
            let path = Path::new(path);
            for query in [
                "MY NOTES.MD",
                "my notes",
                " \"My Notes.md\" ",
                "`my notes`",
                "my people/My Notes.md",
                r"my people\My Notes.md",
            ] {
                assert!(names_file(query, path), "{query}: {path:?}");
            }
            for query in [
                "notes.md",
                "my notes today",
                "people/My Notes.md",
                r"people\My Notes.md",
                "  ``  ",
            ] {
                assert!(!names_file(query, path), "{query}: {path:?}");
            }
        }
        assert!(names_file(
            r"people\father.md",
            Path::new("/book/people/father.md")
        ));
        assert!(names_file(
            "people/father.md",
            Path::new(r"C:\book\people\father.md")
        ));
        assert!(!names_file("my father", Path::new(r"C:\book\father.md")));
    }

    #[test]
    fn fusion_keeps_best_score_and_its_query() {
        let s = |id, score| Scored { id, score };
        let fused = fuse_max(
            [vec![s(1, 0.9), s(2, 0.5)], vec![s(2, 0.8), s(3, 0.7)]].into_iter(),
            2,
        );
        let got: Vec<_> = fused
            .iter()
            .map(|f| (f.scored.id, f.scored.score, f.query_index))
            .collect();
        assert_eq!(got, [(1, 0.9, 0), (2, 0.8, 1)]);
    }

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
    fn link_targets_are_not_embedded() {
        assert_eq!(
            strip_link_targets("Sources: [main dossier](../people/Ada.md), ![map](img/map.png)."),
            "Sources: main dossier, !map."
        );
        assert_eq!(
            strip_link_targets("see [[Ada - old]] and [[Marlow|him]]"),
            "see Ada - old and him"
        );
        assert_eq!(
            strip_link_targets("an [unclosed bracket and [x] plain"),
            "an [unclosed bracket and [x] plain"
        );
        assert_eq!(
            strip_link_targets("[spaced](not a url)"),
            "[spaced](not a url)"
        );
    }

    #[test]
    fn input_hash_is_unambiguous() {
        assert_ne!(
            input_hash(&["text", "ab", "c"]),
            input_hash(&["text", "a", "bc"])
        );
    }
}

#[cfg(test)]
mod pre_split_golden {
    use super::*;

    struct IdentityOnly(crate::ModelIdentity);
    impl Embedder for IdentityOnly {
        fn identity(&self) -> &crate::ModelIdentity {
            &self.0
        }
        fn supports(&self, _: crate::Modality) -> bool {
            true
        }
        fn embed_batch(
            &self,
            _: &[EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<crate::Embedding>> {
            panic!("preparation must not run inference")
        }
    }

    #[test]
    fn production_profile_and_prepared_input_hashes_match_pre_split() {
        // Captured by loading Gemma and calling this engine's build/prepare
        // functions in base 00c1038; see the fixture provenance README.
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/pre_split_preparation.json"))
                .unwrap();
        let identity = serde_json::from_value(golden["profile"]["model"].clone()).unwrap();
        let engine = SearchEngine::builder()
            .embedder(IdentityOnly(identity))
            .build()
            .unwrap();
        assert_eq!(
            serde_json::to_value(engine.profile()).unwrap(),
            golden["profile"]
        );
        assert_eq!(engine.profile_id(), golden["profile_id"].as_str().unwrap());
        for (name, title, kind, bytes) in [
            (
                "titled",
                Some("Guide"),
                ResourceKind::Text(TextFormat::Markdown),
                b"# Setup\n\nRead the [manual](https://example.com/manual) before installing.\n"
                    .as_slice(),
            ),
            (
                "untitled",
                None,
                ResourceKind::Text(TextFormat::Plain),
                b"A short note with no document title.\n".as_slice(),
            ),
            (
                "image",
                Some("diagram"),
                ResourceKind::Image,
                b"\x89PNG\r\n\x1a\nsynthetic golden bytes".as_slice(),
            ),
        ] {
            let resource = ProvidedResource {
                id: format!("test://golden/{name}").parse().unwrap(),
                kind,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                version: ResourceVersion::unknown(),
                title: title.map(String::from),
            };
            let (jobs, _) = engine.prepare(&resource, bytes.to_vec()).unwrap();
            assert_eq!(jobs.len(), 1);
            assert_eq!(
                jobs[0].segment.input_hash,
                golden["input_hashes"][name].as_str().unwrap(),
                "{name}"
            );
        }
    }
    #[cfg(feature = "pdf")]
    #[test]
    fn preview_decodes_a_budget_rejected_page_once_and_keeps_smaller_pages_available() {
        use crate::pdf::fixture::{Kind, pdf};

        let dir = tempfile::tempdir().unwrap();
        let fixture = |name: &str, texts: &[&str]| {
            let path = dir.path().join(name);
            let source = ResourceId::from_path(&path).unwrap();
            let bytes = pdf(texts, Kind::Plain);
            std::fs::write(&path, &bytes).unwrap();
            let meta = ResourceMetadata {
                media_type: Some("application/pdf".into()),
                modality: crate::Modality::Document,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                content_hash: blake3::hash(&bytes).to_hex().to_string(),
            };
            let pages = crate::pdf::read_pages(&bytes, &source).unwrap();
            (source, meta, pages)
        };
        let (large, large_meta, large_pages) = fixture(
            "large.pdf",
            &["This page exceeds the remaining preview budget.", "basil"],
        );
        let (small, small_meta, small_pages) = fixture("small.pdf", &["mint"]);
        let locator = |page: &crate::pdf::PageText| ResourceLocator::PageText {
            page: page.page,
            byte_start: 0,
            byte_end: page.text.len() as u64,
            text_version: crate::pdf::TEXT_VERSION.into(),
            page_text_hash: blake3::hash(page.text.as_bytes()).to_hex().to_string(),
        };
        // Simulate earlier admitted pages leaving room for just the two small pages.
        let remaining = large_pages[1].text.len() + small_pages[0].text.len();
        assert!(large_pages[0].text.len() > remaining);
        let initial_bytes = crate::pdf::MAX_DOCUMENT_TEXT_BYTES - remaining;
        let mut reader = PreviewReader {
            page_bytes: initial_bytes,
            ..Default::default()
        };
        for _ in 0..2 {
            assert!(
                reader
                    .preview(
                        &crate::FsProvider,
                        &large,
                        &locator(&large_pages[0]),
                        &large_meta,
                        200
                    )
                    .is_none()
            );
        }
        assert_eq!(reader.page_decodes, 1);
        assert_eq!(reader.page_bytes, initial_bytes);

        // Reject only the page, not the whole resource or the remaining request.
        for (source, meta, page, expected) in [
            (&large, &large_meta, &large_pages[1], "basil"),
            (&small, &small_meta, &small_pages[0], "mint"),
        ] {
            let valid = locator(page);
            let mut invalid = valid.clone();
            if let ResourceLocator::PageText { byte_end, .. } = &mut invalid {
                *byte_end += 1;
            }
            assert!(
                reader
                    .preview(&crate::FsProvider, source, &invalid, meta, 200)
                    .is_none()
            );
            assert_eq!(
                reader
                    .preview(&crate::FsProvider, source, &valid, meta, 200)
                    .as_deref(),
                Some(expected)
            );
        }
        assert_eq!(reader.page_decodes, 3);
        assert_eq!(reader.page_bytes, crate::pdf::MAX_DOCUMENT_TEXT_BYTES);
        assert!(
            reader
                .preview(
                    &crate::FsProvider,
                    &large,
                    &locator(&large_pages[0]),
                    &large_meta,
                    200
                )
                .is_none()
        );
        assert_eq!(reader.page_decodes, 3);
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn preview_rejects_a_valid_page_in_a_corrupt_resource() {
        use crate::pdf::fixture::{Kind, pdf};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.pdf");
        let source = ResourceId::from_path(&path).unwrap();
        let text =
            crate::pdf::read_page(&pdf(&["garden basil", "second"], Kind::Plain), &source, 1)
                .unwrap();
        let locator = ResourceLocator::PageText {
            page: 1,
            byte_start: 0,
            byte_end: text.len() as u64,
            text_version: crate::pdf::TEXT_VERSION.into(),
            page_text_hash: blake3::hash(text.as_bytes()).to_hex().to_string(),
        };
        for kind in [Kind::MissingContents, Kind::TruncatedContents] {
            let bytes = pdf(&["garden basil", "second"], kind);
            std::fs::write(&path, &bytes).unwrap();
            let meta = ResourceMetadata {
                media_type: Some("application/pdf".into()),
                modality: crate::Modality::Document,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                content_hash: blake3::hash(&bytes).to_hex().to_string(),
            };
            assert!(
                PreviewReader::default()
                    .preview(&crate::FsProvider, &source, &locator, &meta, 200)
                    .is_none()
            );
        }
    }
}
