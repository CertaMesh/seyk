//! The search engine: discovery, segmentation, lazy embedding, ranking.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Condvar, Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
};
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

mod image_scan;
pub use image_scan::ImageScan;

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

/// Why a response contains only keyword and filename matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedReason {
    ModelLoading,
    NotYetEmbedded,
    NoModel,
}

/// Additive response metadata, shared by library, CLI JSON and MCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchDegradation {
    pub mode: DegradedMode,
    pub reason: DegradedReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedMode {
    LexicalOnly,
}

impl SearchDegradation {
    pub fn lexical(reason: DegradedReason) -> Self {
        Self {
            mode: DegradedMode::LexicalOnly,
            reason,
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

/// Per-leg positions in the candidate lists used to rank a hit.
/// Ranks are one-based; `None` means the leg did not contribute. These ranks
/// precede fusion and the final result limit, so can exceed that limit.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchReason {
    pub semantic_rank: Option<usize>,
    /// The BM25 leg, called `lexical` in [`MatchSignals`].
    pub keyword_rank: Option<usize>,
    pub filename_rank: Option<usize>,
    /// Normalized query tokens from matching BM25 groups, deduplicated in query
    /// order. Never extracted from source text. Empty without a keyword leg.
    pub matched_terms: Vec<String>,
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
    /// Rank provenance, additive to the existing raw signal scores.
    #[serde(default)]
    pub r#match: MatchReason,
    pub metadata: ResourceMetadata,
    /// Markdown heading path of the segment, when there is one.
    pub heading: Option<String>,
    /// A short excerpt of the matched segment (text resources only).
    pub preview: Option<String>,
    /// Index into [`SearchResults::queries`] of the query that scored best here.
    /// Meaningful only when [`MatchSignals::semantic`] is present.
    pub query_index: usize,
}

/// Stable classification independent of dependency error messages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipCode {
    #[default]
    Other,
    PdfParseFailed,
    PdfParserPanicked,
}

/// Number of resources skipped for one typed reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedCount {
    pub code: SkipCode,
    pub reason: String,
    pub count: usize,
}

/// A resource that was found but could not be searched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedResource {
    #[serde(default)]
    pub code: SkipCode,
    pub resource: ResourceId,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::resource::serialize_optional_path"
    )]
    pub path: Option<PathBuf>,
    pub reason: String,
}

/// A page omitted after a parse or preparation failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageDiagnostic {
    #[serde(default)]
    pub code: SkipCode,
    pub resource: ResourceId,
    pub page: u32,
    pub reason: String,
}

/// What happened while bringing the cache up to date.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncStats {
    /// Current-pass resource inventory (unavailable when synchronization is disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<crate::IndexProgress>,
    pub resources_discovered: usize,
    /// Reused from cache without re-embedding.
    pub resources_unchanged: usize,
    /// New or changed since last seen.
    pub resources_updated: usize,
    pub skipped: Vec<SkippedResource>,
    #[serde(default)]
    pub skipped_counts: Vec<SkippedCount>,
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
    /// Present only when semantic retrieval was unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded: Option<SearchDegradation>,
    pub stats: SearchStats,
    /// Counts of returned hits, after ranking and limit, keyed by locator kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind_counts: Option<std::collections::BTreeMap<String, usize>>,
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
    /// Resource counts for this indexing pass; independent of batch-local segments.
    IndexProgress {
        progress: crate::IndexProgress,
    },
    /// Roots this pass's listing actually walked, resolved once before the walk,
    /// in scope order. Sent just before `Discovered`, and only when the provider
    /// can attribute its listing to them; every later count in the pass then
    /// describes resources found under these roots.
    Walked {
        roots: Vec<ResourceId>,
    },
    Discovered {
        resources: usize,
    },
    /// `total` segments need embedding; `done` so far.
    Embedding {
        done: usize,
        total: usize,
    },
}

/// Called without the cache mutex held. Resource-count and embedding events
/// hold the indexing lease: callbacks may inspect the cache or search with
/// `sync(false)`, but
/// must not synchronously start another indexing pass or call `scrub_cache`
/// on the same engine, since either would reacquire the indexing lease.
pub type ProgressFn<'a> = &'a (dyn Fn(ProgressEvent) + Sync);

fn report_walked(progress: Option<ProgressFn<'_>>, roots: Option<Vec<ResourceId>>) {
    if let (Some(p), Some(roots)) = (progress, roots) {
        p(ProgressEvent::Walked { roots });
    }
}

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

    /// Describe vector identity without constructing an engine or opening a cache.
    pub fn cache_profile(&self, model: crate::embedder::ModelIdentity) -> CacheProfile {
        CacheProfile {
            model,
            segmenter_version: format!(
                "{}+input-v{INPUT_FORMAT_VERSION}",
                self.segmenter.version()
            ),
        }
    }

    pub fn build(mut self) -> Result<SearchEngine> {
        crate::discovery::audio_byte_limit(self.max_audio_duration_secs)?;
        #[cfg(feature = "pdf")]
        self.extractors.push(Arc::new(crate::PdfTextExtractor));
        self.extractors.push(Arc::new(TextExtractor));
        let embedder = self
            .embedder
            .take()
            .ok_or_else(|| Error::Config("SearchEngine needs an embedder".into()))?;
        let profile = self.cache_profile(embedder.identity().clone());
        let profile_id = profile.id();
        let mut cache = self.cache.unwrap_or_else(|| Box::new(MemoryCache::new()));
        cache.register_profile(&profile_id, &profile)?;
        Ok(SearchEngine {
            provider: self.provider,
            extractors: self.extractors,
            #[cfg(feature = "audio")]
            audio_extractor: self.audio_extractor,
            embedder,
            lexical_only: false,
            skipped_versions: Mutex::new(HashMap::new()),
            cache: Arc::new(Mutex::new(cache)),
            indexing: Arc::new(Mutex::new(())),
            pdf_diagnostics: Mutex::new(HashMap::new()),
            scrubber_started: AtomicBool::new(false),
            #[cfg(all(feature = "test-support", debug_assertions))]
            lease_observer: Mutex::new(None),
            scrubber_stop: Arc::new(ScrubStop::default()),
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
    lexical_only: bool,
    /// Completed skips for this engine lifetime, keyed by source version and
    /// preparation descriptor. A restart retries them during the initial index;
    /// no failure state or source content is added to the semantic cache schema.
    skipped_versions: Mutex<HashMap<ResourceId, (ResourceVersion, String)>>,
    cache: Arc<Mutex<Box<dyn EmbeddingCache>>>,
    /// Acquire before `cache`, never while holding it. Covers pending vectors
    /// until their resource references are committed and pruning is complete.
    indexing: Arc<Mutex<()>>,
    pdf_diagnostics: Mutex<HashMap<(ResourceId, String), PdfDiagnostics>>,
    scrubber_started: AtomicBool,
    #[cfg(all(feature = "test-support", debug_assertions))]
    lease_observer: Mutex<Option<LeaseObserver>>,
    scrubber_stop: Arc<ScrubStop>,
    segmenter: TextSegmenter,
    profile: CacheProfile,
    profile_id: String,
    batch_size: usize,
    max_audio_duration_secs: u64,
}

/// Test-only observations at the indexing lease boundary.
#[cfg(all(feature = "test-support", debug_assertions))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseEvent {
    /// A real try_lock failed while another pass held the lease.
    Contended,
    Acquired,
}

#[cfg(all(feature = "test-support", debug_assertions))]
type LeaseObserver = Arc<dyn Fn(LeaseEvent) + Send + Sync>;
// Shared only by the engine and worker; contains no engine/cache references.
#[derive(Default)]
struct ScrubStop {
    state: Mutex<ScrubStopState>,
    wake: Condvar,
}

#[derive(Default)]
struct ScrubStopState {
    stopped: bool,
    #[cfg(test)]
    waiting: bool,
}

impl ScrubStop {
    fn wait(&self, delay: std::time::Duration) -> bool {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        #[cfg(test)]
        let state = {
            let mut state = state;
            state.waiting = true;
            self.wake.notify_all();
            state
        };
        let Ok((state, _)) = self.wake.wait_timeout_while(state, delay, |s| !s.stopped) else {
            return false;
        };
        #[cfg(test)]
        let state = {
            let mut state = state;
            state.waiting = false;
            state
        };
        !state.stopped
    }

    fn is_stopped(&self) -> bool {
        self.state.lock().map_or(true, |state| state.stopped)
    }

    fn stop(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.stopped = true;
        self.wake.notify_all();
    }
}

impl Drop for SearchEngine {
    fn drop(&mut self) {
        self.scrubber_stop.stop();
    }
}

impl SearchEngine {
    /// Install a test observer. It must not start indexing on this engine.
    #[cfg(all(feature = "test-support", debug_assertions))]
    pub fn observe_indexing_lease(&self, observer: LeaseObserver) {
        *self
            .lease_observer
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(observer);
    }

    #[cfg(all(feature = "test-support", debug_assertions))]
    fn observe_lease(&self, event: LeaseEvent) {
        let observer = self
            .lease_observer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(observer) = observer {
            observer(event);
        }
    }

    fn acquire_indexing_lease(&self) -> Result<MutexGuard<'_, ()>> {
        // One mutex path in every build; only observations are test-specific.
        let guard = match self.indexing.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => {
                #[cfg(all(feature = "test-support", debug_assertions))]
                self.observe_lease(LeaseEvent::Contended);
                self.indexing
                    .lock()
                    .map_err(|_| Error::Config("indexing lock poisoned".into()))?
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(Error::Config("indexing lock poisoned".into()));
            }
        };
        #[cfg(all(feature = "test-support", debug_assertions))]
        self.observe_lease(LeaseEvent::Acquired);
        Ok(guard)
    }

    /// Search text and PDF text without model weights or stored vectors.
    /// Uses an isolated in-memory keyword index; never downloads a model.
    pub fn lexical() -> Result<Self> {
        Self::lexical_with_provider(Arc::new(FsProvider))
    }

    /// Model-free keyword search using the same access policy as the caller.
    pub fn lexical_with_provider(provider: Arc<dyn ResourceProvider>) -> Result<Self> {
        let mut engine = Self::builder()
            .shared_provider(provider)
            .embedder(crate::lexical_engine::LexicalIdentity::default())
            .build()?;
        engine.lexical_only = true;
        Ok(engine)
    }

    fn lexical_fallback(&self) -> Result<Self> {
        let mut engine = Self::lexical_with_provider(self.provider.clone())?;
        engine.segmenter = self.segmenter.clone();
        engine.max_audio_duration_secs = self.max_audio_duration_secs;
        Ok(engine)
    }

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
    fn audio_descriptor_hash(&self, resource: &ProvidedResource) -> Result<String> {
        #[cfg(feature = "audio")]
        {
            Ok(input_hash(&[
                &resource.fingerprint(),
                crate::audio::PREPARATION,
                self.embedder.input_identity(crate::Modality::Audio)?,
            ]))
        }
        #[cfg(not(feature = "audio"))]
        {
            Ok(resource.fingerprint())
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
            image_scan: ImageScan::Exact,
            lexical_weight: None,
            lexical_fallback: false,
            kind_counts: false,
            listed: None,
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
    /// The closure must not start indexing or call `scrub_cache`: both acquire
    /// the indexing lease before the cache mutex and would self-deadlock.
    pub fn with_cache<T>(&self, f: impl FnOnce(&mut dyn EmbeddingCache) -> Result<T>) -> Result<T> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| Error::Config("cache lock poisoned".into()))?;
        f(guard.as_mut())
    }

    /// Explicit privacy scrub. Index/search requests never invoke this themselves.
    ///
    /// Do not call this from `with_cache` closures or indexing progress callbacks
    /// on this engine: reacquiring the cache mutex or indexing lease would deadlock.
    pub fn scrub_cache(&self) -> Result<crate::cache::CacheCleanup> {
        let _lease = self
            .indexing
            .lock()
            .map_err(|_| Error::Config("indexing lock poisoned".into()))?;
        self.with_cache(|cache| cache.scrub())
    }

    /// Start one idle-time worker for watcher/daemon/MCP use. It holds weak
    /// references and never runs in a request. On engine drop it stops after its
    /// current iteration and wakes any idle deadline wait immediately. Drop does
    /// not cancel or join an in-flight scrub.
    pub fn start_cache_scrubber(&self) {
        self.spawn_cache_scrubber();
    }

    fn spawn_cache_scrubber(&self) -> Option<std::thread::JoinHandle<()>> {
        let cache = Arc::downgrade(&self.cache);
        let stop = self.scrubber_stop.clone();
        self.start_cache_scrubber_with(
            move || {
                let delay = {
                    let Some(cache) = cache.upgrade() else {
                        return false;
                    };
                    let Ok(cache) = cache.lock() else {
                        return false;
                    };
                    cache.scrub_delay(std::time::Duration::from_secs(60))
                }; // Release strong references and the mutex before sleeping.
                stop.wait(delay)
            },
            |cache| match cache.scrub_if_idle(std::time::Duration::from_secs(60)) {
                Ok(Some(report)) if report.pending => eprintln!(
                    "warning: cache scrub pending; run seyk cache scrub once other readers close"
                ),
                Err(error) => eprintln!("warning: cache scrub pending: {error}"),
                _ => {}
            },
        )
    }

    // Injectable ticks and work make worker ownership/locking tests deterministic.
    // Production sleeps to the quiet/retry deadline (60 seconds when idle).
    fn start_cache_scrubber_with(
        &self,
        mut tick: impl FnMut() -> bool + Send + 'static,
        mut scrub: impl FnMut(&mut dyn EmbeddingCache) + Send + 'static,
    ) -> Option<std::thread::JoinHandle<()>> {
        if self.scrubber_started.swap(true, Ordering::Relaxed) {
            return None;
        }
        let cache = Arc::downgrade(&self.cache);
        let indexing = Arc::downgrade(&self.indexing);
        let stop = self.scrubber_stop.clone();
        Some(std::thread::spawn(move || {
            while !stop.is_stopped() && tick() {
                if stop.is_stopped() {
                    break;
                }
                let (Some(cache), Some(indexing)) = (cache.upgrade(), indexing.upgrade()) else {
                    break;
                };
                // Do not scrub in the middle of an indexing pass, even a slow embed batch.
                let Ok(_lease) = indexing.try_lock() else {
                    continue;
                };
                let Ok(mut cache) = cache.lock() else { break };
                scrub(cache.as_mut());
            }
        }))
    }

    /// Make sure every resource has up-to-date vectors in the cache.
    /// The caller keeps the indexing lease through any subsequent deletions.
    fn sync(
        &self,
        _indexing: &MutexGuard<'_, ()>,
        resources: &[ProvidedResource],
        progress: Option<ProgressFn>,
    ) -> Result<SyncStats> {
        let stats = self.sync_resources(resources, progress)?;
        let skipped: HashSet<_> = stats.skipped.iter().map(|s| &s.resource).collect();
        // A descriptor can depend on model identity (audio). If identity itself
        // failed, there is no stable preparation version to remember: the normal
        // eligibility check must be allowed to retry when weights are restored.
        let outcomes: Vec<_> = resources
            .iter()
            .map(|r| {
                let outcome = if skipped.contains(&r.id) {
                    self.preparation_descriptor(r)
                        .ok()
                        .map(|descriptor| (r.version.clone(), descriptor))
                } else {
                    None
                };
                (r.id.clone(), outcome)
            })
            .collect();
        let mut remembered = self
            .skipped_versions
            .lock()
            .map_err(|_| Error::Config("skip outcomes lock poisoned".into()))?;
        for (id, outcome) in outcomes {
            if let Some(outcome) = outcome {
                remembered.insert(id, outcome);
            } else {
                remembered.remove(&id);
            }
        }
        Ok(stats)
    }

    fn skipped_at_current_version(&self, resource: &ProvidedResource) -> Result<bool> {
        let outcome = self
            .skipped_versions
            .lock()
            .map_err(|_| Error::Config("skip outcomes lock poisoned".into()))?
            .get(&resource.id)
            .cloned();
        Ok(outcome.is_some_and(|(version, descriptor)| {
            version.matches(&resource.version)
                && self
                    .preparation_descriptor(resource)
                    .is_ok_and(|current| current == descriptor)
        }))
    }

    fn preparation_descriptor(&self, r: &ProvidedResource) -> Result<String> {
        if r.kind == ResourceKind::Audio {
            return self.audio_descriptor_hash(r);
        }
        #[cfg(feature = "pdf")]
        if r.kind == ResourceKind::Pdf {
            return Ok(input_hash(&[
                &r.fingerprint(),
                crate::pdf::TEXT_VERSION,
                crate::pdf::PREPARATION_VERSION,
                &self.pdf_capabilities(),
            ]));
        }
        Ok(r.fingerprint())
    }

    fn sync_resources(
        &self,
        resources: &[ProvidedResource],
        progress: Option<ProgressFn>,
    ) -> Result<SyncStats> {
        let mut stats = SyncStats {
            resources_discovered: resources.len(),
            ..Default::default()
        };
        let pid = self.profile_id.as_str();
        let known = self.with_cache(|c| c.resource_index(pid))?;

        let telemetry = crate::progress::PassProgress::new(resources, progress);
        let mut pending: Vec<Pending> = Vec::new();
        let mut reported_skips = 0;
        for r in resources {
            for skipped in &stats.skipped[reported_skips..] {
                telemetry.complete(&skipped.resource, false, false);
            }
            reported_skips = stats.skipped.len();
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
            let descriptor_hash = match self.preparation_descriptor(r) {
                Ok(hash) => hash,
                Err(error) => {
                    stats.skipped.push(skip(&r.id, error.to_string()));
                    continue;
                }
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
                    self.replay_pdf_diagnostics(&r.id, &descriptor_hash, &mut stats)?;
                    telemetry.complete(&r.id, stats.skipped.len() == reported_skips, false);
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
                    telemetry.complete(&r.id, true, false);
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
                } else {
                    telemetry.complete(&r.id, true, true);
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
                self.replay_pdf_diagnostics(&r.id, &descriptor_hash, &mut stats)?;
                telemetry.complete(&r.id, stats.skipped.len() == reported_skips, false);
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
                if let Some(entry) = self.sync_pdf(r, bytes, target, &mut stats, progress)? {
                    stats.skipped.push(entry);
                } else {
                    telemetry.complete(&r.id, true, true);
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

        for skipped in &stats.skipped[reported_skips..] {
            telemetry.complete(&skipped.resource, false, false);
        }

        if self.lexical_only {
            self.write_resources(
                &pending,
                0,
                pending.len(),
                &HashMap::new(),
                &mut stats,
                &telemetry,
            )?;
            stats.progress = Some(telemetry.snapshot());
            stats.skipped_counts = skipped_counts(&stats.skipped);
            return Ok(stats);
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
            written = self.write_resources(
                &pending, written, ready_upto, &failed, &mut stats, &telemetry,
            )?;
        }
        self.write_resources(
            &pending,
            written,
            pending.len(),
            &failed,
            &mut stats,
            &telemetry,
        )?;
        stats.embed_ms += started.elapsed().as_millis() as u64;
        if stats.resources_updated > 0 {
            self.with_cache(|c| c.prune_orphans(pid))?;
        }
        for skipped in &stats.skipped {
            telemetry.complete(&skipped.resource, false, false);
        }
        stats.progress = Some(telemetry.snapshot());
        stats.skipped_counts = skipped_counts(&stats.skipped);
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
        telemetry: &crate::progress::PassProgress<'_>,
    ) -> Result<usize> {
        let pid = self.profile_id.as_str();
        for p in pending.iter().take(upto).skip(from) {
            if let Some(reason) = p
                .jobs
                .iter()
                .find_map(|j| failed.get(&j.segment.input_hash))
            {
                stats.skipped.push(skip(&p.resource, reason.clone()));
                telemetry.complete(&p.resource, false, false);
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
            telemetry.complete(&p.resource, true, true);
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

    fn replay_pdf_diagnostics(
        &self,
        resource: &ResourceId,
        descriptor: &str,
        stats: &mut SyncStats,
    ) -> Result<()> {
        let diagnostics = self
            .pdf_diagnostics
            .lock()
            .map_err(|_| Error::Config("PDF diagnostics lock poisoned".into()))?;
        if let Some(record) = diagnostics.get(&(resource.clone(), descriptor.to_owned())) {
            stats.page_diagnostics.extend(record.pages.clone());
            if let Some(skipped) = &record.skipped {
                stats.skipped.push(skipped.clone());
            }
        }
        Ok(())
    }

    #[cfg(feature = "pdf")]
    fn remember_pdf_diagnostics(
        &self,
        target: &Pending,
        skipped: Option<SkippedResource>,
    ) -> Result<()> {
        let mut diagnostics = self
            .pdf_diagnostics
            .lock()
            .map_err(|_| Error::Config("PDF diagnostics lock poisoned".into()))?;
        diagnostics.retain(|(resource, _), _| resource != &target.resource);
        if !target.page_diagnostics.is_empty() || skipped.is_some() {
            diagnostics.insert(
                (target.resource.clone(), target.descriptor_hash.clone()),
                PdfDiagnostics {
                    pages: target.page_diagnostics.clone(),
                    skipped,
                },
            );
        }
        Ok(())
    }

    #[cfg(feature = "pdf")]
    fn sync_pdf(
        &self,
        r: &ProvidedResource,
        bytes: Vec<u8>,
        mut target: Pending,
        stats: &mut SyncStats,
        progress: Option<ProgressFn>,
    ) -> Result<Option<SkippedResource>> {
        let pages = match crate::pdf::read_pages(&bytes, &r.id) {
            Ok(pages) => pages,
            Err(error) => {
                let entry = pdf_skip(&r.id, error);
                self.commit_media_resource(&target, &[], stats)?;
                self.remember_pdf_diagnostics(&target, Some(entry.clone()))?;
                return Ok(Some(entry));
            }
        };
        // Failed pages never shift later page numbers or become rendered text.
        // If every page failed, retain resource-level failure semantics as well.
        let all_failed = pages.iter().all(|page| page.error.is_some());
        let mut first_failure = None;
        let mut valid_pages = Vec::new();
        for page in pages {
            if let Some(error) = page.error {
                let entry = pdf_skip(&r.id, error);
                let reason = entry.reason.clone();
                target.page_diagnostics.push(PageDiagnostic {
                    code: entry.code,
                    resource: r.id.clone(),
                    page: page.page,
                    reason: reason.clone(),
                });
                first_failure.get_or_insert(entry);
            } else {
                valid_pages.push(page);
            }
        }
        if all_failed {
            self.commit_media_resource(&target, &[], stats)?;
            self.remember_pdf_diagnostics(&target, first_failure.clone())?;
            stats.page_diagnostics.extend(target.page_diagnostics);
            return Ok(first_failure);
        }
        #[cfg(feature = "pdf-render")]
        let mut renderer = None;
        let capacity = self.batch_size.min(4);
        let mut batch = Vec::with_capacity(capacity);
        let mut records = Vec::new();
        for page in valid_pages {
            if crate::pdf::usable(&page.text) {
                if !self.embedder.supports(crate::Modality::Text) {
                    target.page_diagnostics.push(PageDiagnostic {
                        code: SkipCode::Other,
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
                        return Ok(Some(skip(&r.id, reason)));
                    }
                }
            } else if self.pdf_images() {
                #[cfg(feature = "pdf-render")]
                {
                    if renderer.is_none() {
                        renderer = match crate::pdf::render::PageRenderer::new(&bytes, &r.id) {
                            Ok(renderer) => Some(renderer),
                            Err(error) => return Ok(Some(pdf_skip(&r.id, error))),
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
                                code: SkipCode::Other,
                                resource: r.id.clone(),
                                page: page.page,
                                reason: reason.into(),
                            });
                            continue;
                        }
                        Err(error) => return Ok(Some(pdf_skip(&r.id, error))),
                    }
                }
            } else {
                target.page_diagnostics.push(PageDiagnostic {
                    code: SkipCode::Other,
                    resource: r.id.clone(),
                    page: page.page,
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
                return Ok(Some(skip(&r.id, reason)));
            }
        }
        if let Some(reason) = self.flush_media_batch(&mut batch, &mut records, stats, progress)? {
            return Ok(Some(skip(&r.id, reason)));
        }
        self.commit_media_resource(&target, &records, stats)?;
        self.remember_pdf_diagnostics(&target, None)?;
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
        if self.lexical_only {
            records.extend(batch.drain(..).map(|j| (j.segment, j.lexical_body)));
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
        let identity = self.embedder.input_identity(crate::Modality::Audio)?;
        let result = crate::audio::windows_verified(
            source,
            &r.id,
            self.max_audio_duration_secs,
            |locator, samples| {
                let mut hash = blake3::Hasher::new();
                hash.update(b"audio-input-v1");
                hash.update(crate::audio::PREPARATION.as_bytes());
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
fn pdf_skip(resource: &ResourceId, error: Error) -> SkippedResource {
    let code = match &error {
        Error::ParserPanicked { .. } => SkipCode::PdfParserPanicked,
        Error::PdfParse { .. } | Error::PdfPageDecode { .. } => SkipCode::PdfParseFailed,
        _ => SkipCode::Other,
    };
    let reason = match error {
        Error::PdfParse { reason, .. } | Error::PdfPageDecode { reason, .. } => {
            format!("PDF parse failed: {reason}")
        }
        Error::PdfRender { reason, .. } => format!("PDF rendering failed: {reason}"),
        Error::ParserPanicked { reason, .. } => format!("PDF parser panicked: {reason}"),
        other => other.to_string(),
    };
    SkippedResource {
        code,
        resource: resource.clone(),
        path: resource.to_file_path(),
        reason,
    }
}

struct PdfDiagnostics {
    pages: Vec<PageDiagnostic>,
    skipped: Option<SkippedResource>,
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
    let reason = reason.into();
    SkippedResource {
        code: SkipCode::Other,
        resource: resource.clone(),
        path: resource.to_file_path(),
        reason,
    }
}

fn skipped_counts(skipped: &[SkippedResource]) -> Vec<SkippedCount> {
    let mut counts = std::collections::BTreeMap::new();
    for entry in skipped {
        *counts
            .entry((entry.code, entry.reason.clone()))
            .or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|((code, reason), count)| SkippedCount {
            code,
            reason,
            count,
        })
        .collect()
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
    image_scan: ImageScan,
    lexical_weight: Option<f32>,
    lexical_fallback: bool,
    kind_counts: bool,
    /// Resources the outer request already listed and authorized. A nested
    /// fallback prepares exactly these, so it never widens scope or re-walks.
    listed: Option<Vec<ProvidedResource>>,
}

impl<'e> SearchRequest<'e> {
    /// Opt into experimental Photo Anchor candidate scanning for standalone images.
    /// Exact f32 reranking preserves returned score semantics. Requires SQLite;
    /// a bounded shortlist does not guarantee recall on arbitrary libraries.
    pub fn image_scan(mut self, scan: ImageScan) -> Self {
        self.image_scan = scan;
        self
    }

    /// Include counts of returned hits by locator kind; no second search is run.
    pub fn kind_counts(mut self, enabled: bool) -> Self {
        self.kind_counts = enabled;
        self
    }

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

    /// Allow a cached search to prepare an ephemeral keyword index when any
    /// supported resource is new or changed. Never waits for embedding or writes
    /// to the semantic cache. Providers opting into cached listing retain their
    /// indexed-only scope; their background indexer discovers new resources.
    /// Without this opt-in, `sync(false)` stays cache-only.
    pub fn lexical_fallback(mut self, enabled: bool) -> Self {
        self.lexical_fallback = enabled;
        self
    }

    pub fn execute(mut self) -> Result<SearchResults> {
        let started = Instant::now();
        let engine = self.engine;
        if engine.lexical_only {
            self.sync = true;
            self.mode = SearchMode::Lexical;
        }
        if self.image_scan == ImageScan::PhotoAnchorQ4 {
            if self.mode == SearchMode::Lexical {
                return Err(Error::Config(
                    "q4 image scan is unavailable in lexical-only mode".into(),
                ));
            }
            if !engine.with_cache(|c| Ok(c.supports_image_codes()))? {
                return Err(Error::Config(
                    "q4 image scan requires SQLite; --no-cache is unsupported".into(),
                ));
            }
            crate::cache::image_codes::check_profile(&engine.profile)?;
        }
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
        let mut discovery = self.discovery.clone();
        discovery.max_audio_duration_secs = engine.max_audio_duration_secs;
        let scope = ResourceScope::from_paths(&self.paths, self.roots.clone(), discovery)?;
        let pid = engine.profile_id.as_str();
        // Only opted-in providers authorize the indexed subset without walking
        // uncached descendants. Default providers retain their original list path.
        let listed = self.listed.take();
        let cached_metadata =
            if listed.is_none() && !self.sync && engine.provider.supports_cached_listing() {
                Some(engine.with_cache(|c| c.resource_index(pid))?)
            } else {
                None
            };
        let resources = if let Some(listed) = listed {
            // The outer request already reported Walked/Discovered.
            listed
        } else {
            let resources = if let Some(metadata) = &cached_metadata {
                let ids: Vec<_> = metadata.keys().cloned().collect();
                engine.provider.list_cached(&scope, &ids)?
            } else {
                let listing = engine.provider.list_walked(&scope)?;
                report_walked(self.progress, listing.walked_roots);
                listing.resources
            };
            if let Some(p) = self.progress {
                p(ProgressEvent::Discovered {
                    resources: resources.len(),
                });
            }
            resources
        };
        if self.lexical_fallback && !self.sync && !engine.lexical_only {
            let cached = engine.with_cache(|c| c.resource_index(&engine.profile_id))?;
            let mut pending = false;
            for r in &resources {
                if engine.supports_kind(r.kind)
                    && !engine.skipped_at_current_version(r)?
                    && !cached
                        .get(&r.id)
                        .is_some_and(|m| !r.version.is_known() || m.version.matches(&r.version))
                {
                    pending = true;
                    break;
                }
            }
            if pending {
                let lexical = engine.lexical_fallback()?;
                let mut results = SearchRequest {
                    engine: &lexical,
                    sync: true,
                    image_scan: ImageScan::Exact,
                    listed: Some(resources),
                    ..self
                }
                .execute()?;
                results.degraded = Some(SearchDegradation::lexical(DegradedReason::NotYetEmbedded));
                results.stats.total_ms = started.elapsed().as_millis() as u64;
                return Ok(results);
            }
        }
        let mut sync = if self.sync {
            let indexing = engine.acquire_indexing_lease()?;
            let report = engine.sync(&indexing, &resources, self.progress)?;
            if self.image_scan == ImageScan::PhotoAnchorQ4 {
                let skipped: HashSet<_> = report.skipped.iter().map(|s| &s.resource).collect();
                let paths = resources
                    .iter()
                    .filter(|r| r.kind == ResourceKind::Image && !skipped.contains(&r.id))
                    .map(|r| r.id.clone())
                    .collect();
                engine.with_cache(|c| c.sync_image_codes(pid, &paths))?;
            }
            report
        } else {
            SyncStats {
                resources_discovered: resources.len(),
                ..Default::default()
            }
        };

        let mut wanted: HashSet<ResourceId> = resources
            .iter()
            .filter(|r| engine.supports_kind(r.kind))
            .map(|r| r.id.clone())
            .collect();
        let metadata = match cached_metadata {
            Some(metadata) => metadata,
            None => engine.with_cache(|c| c.resource_index(pid))?,
        };
        // Failed preparation must never expose a previous version's evidence,
        // including filename-only hits and requests without previews. Apply this
        // to normal synchronized searches as well as the model-free engine.
        let skipped: HashSet<_> = sync.skipped.iter().map(|s| s.resource.clone()).collect();
        for r in &resources {
            if skipped.contains(&r.id)
                || engine.skipped_at_current_version(r)?
                || (self.lexical_fallback
                    && r.version.is_known()
                    && !metadata
                        .get(&r.id)
                        .is_some_and(|m| m.version.matches(&r.version)))
            {
                wanted.remove(&r.id);
            }
        }
        for r in resources.iter().filter(|r| r.kind == ResourceKind::Audio) {
            let failed = skipped.contains(&r.id);
            if failed {
                wanted.remove(&r.id);
                continue;
            }
            let descriptor = match engine.audio_descriptor_hash(r) {
                Ok(hash) => hash,
                Err(error) => {
                    sync.skipped.push(skip(&r.id, error.to_string()));
                    wanted.remove(&r.id);
                    continue;
                }
            };
            let current = metadata.get(&r.id).is_some_and(|m| {
                // Unknown/changed cached sources cannot establish eligibility in cached-only search.
                (self.sync || m.version.matches(&r.version))
                    && m.metadata.size_bytes == r.size_bytes
                    && m.descriptor_hash.as_ref() == Some(&descriptor)
            });
            if !current || !engine.audio_eligible(&r.id)? {
                wanted.remove(&r.id);
            }
        }

        sync.skipped_counts = skipped_counts(&sync.skipped);

        // Each signal contributes a deeper list than asked for, so a hit that is
        // only moderately good on two signals can still surface.
        let depth = self.limit.saturating_mul(4).max(50);

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
            let min_score = self.min_score.unwrap_or(f32::MIN);
            let take = if self.mode == SearchMode::Semantic {
                self.limit
            } else {
                depth
            };
            if self.image_scan == ImageScan::PhotoAnchorQ4 {
                let mut rows = engine.with_cache(|c| c.image_scan_rows(pid, &wanted))?;
                rows.retain(|v| {
                    metadata
                        .get(&v.resource)
                        .is_some_and(|m| engine.supports_locator(&m.metadata, &v.segment.locator))
                });
                rows.sort_by_cached_key(|v| (crate::cache::resource_key(&v.resource), v.ordinal));
                searched = rows.len();
                let ranked = image_scan::search(engine, &rows, &query_vectors, take, min_score)?;
                let keys: Vec<_> = rows
                    .into_iter()
                    .map(|v| segments.insert(v.resource, v.ordinal, v.segment))
                    .collect();
                semantic = fuse_max(ranked.into_iter(), take)
                    .into_iter()
                    .map(|f| (keys[f.scored.id].clone(), f.scored.score, f.query_index))
                    .collect();
            } else {
                let mut stored = engine.with_cache(|c| c.vectors_for(pid, &wanted))?;
                // Vector IDs break cosine ties, including at the top-k cutoff.
                stored.sort_by_cached_key(|v| (crate::cache::resource_key(&v.resource), v.ordinal));
                let mut index = BruteForceIndex::with_capacity(dim, stored.len())?;
                let mut keys = Vec::with_capacity(stored.len());
                for v in stored {
                    // A wrong-sized vector means a corrupt cache; never compare it.
                    if v.vector.len() == dim
                        && metadata.get(&v.resource).is_some_and(|m| {
                            engine.supports_locator(&m.metadata, &v.segment.locator)
                        })
                    {
                        index.add(&v.vector);
                        keys.push(segments.insert(v.resource, v.ordinal, v.segment));
                    }
                }
                searched = keys.len();
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

        // Annotate only after ranking. Keep the scoring arithmetic, candidate
        // order, cutoffs and tie-breaking completely independent of provenance.
        let semantic_ranks: HashMap<_, _> = semantic
            .iter()
            .enumerate()
            .map(|(i, (key, _, _))| (key, i + 1))
            .collect();
        let keyword_ranks: HashMap<_, _> = lexical
            .iter()
            .enumerate()
            .map(|(i, (key, _))| (key, i + 1))
            .collect();
        let filename_ranks: HashMap<_, _> = named
            .iter()
            .enumerate()
            .map(|(i, key)| (key, i + 1))
            .collect();
        let mut previews = PreviewReader::default();
        let mut hits = Vec::with_capacity(ranked.len());
        let mut keyword_keys = Vec::new();
        let mut keyword_positions = Vec::new();
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
            let keyword_rank = signals
                .lexical
                .and_then(|_| keyword_ranks.get(&key).copied());
            let reason = MatchReason {
                semantic_rank: signals
                    .semantic
                    .and_then(|_| semantic_ranks.get(&key).copied()),
                keyword_rank,
                filename_rank: signals.filename.then(|| filename_ranks[&key]),
                matched_terms: Vec::new(),
            };
            if keyword_rank.is_some() {
                keyword_keys.push(key.clone());
                keyword_positions.push(hits.len());
            }
            hits.push(SearchHit {
                resource: key.0,
                locator: segment.locator.clone(),
                score,
                signals,
                r#match: reason,
                metadata,
                heading: segment.heading.clone(),
                preview,
                query_index,
            });
        }

        if !keyword_keys.is_empty() {
            let query = LexicalQuery::new(self.queries.iter().map(String::as_str));
            let terms =
                engine.with_cache(|c| c.lexical_matched_terms(pid, &query, &keyword_keys))?;
            for (position, terms) in keyword_positions.into_iter().zip(terms) {
                hits[position].r#match.matched_terms = terms;
            }
        }

        let kind_counts = self.kind_counts.then(|| {
            let mut counts = std::collections::BTreeMap::new();
            for hit in &hits {
                *counts.entry(hit.locator.kind().to_owned()).or_insert(0) += 1;
            }
            counts
        });
        Ok(SearchResults {
            kind_counts,
            queries: self.queries,
            mode: self.mode,
            hits,
            degraded: engine
                .lexical_only
                .then(|| SearchDegradation::lexical(DegradedReason::NoModel)),
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

/// Reciprocal rank fusion damping. A full SciFact/NFCorpus/IRPAPERS sweep
/// improved hybrid nDCG@10 on all three datasets with k=5 versus k=60,
/// keeping lexical weights, filename weight and candidate depth unchanged.
const RRF_K: f32 = 5.0;
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
        let listing = engine.provider.list_walked(&scope)?;
        report_walked(self.progress, listing.walked_roots);
        let resources = listing.resources;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let indexing = engine.acquire_indexing_lease()?;
        let mut sync = engine.sync(&indexing, &resources, self.progress)?;

        let pid = engine.profile_id.as_str();
        let present: HashSet<&ResourceId> = resources.iter().map(|r| &r.id).collect();
        let known = engine.with_cache(|c| c.resource_index(pid))?;
        // A skipped resource may never have acquired a cache row. Include those
        // identities in the same conservative, scope-aware deletion checks.
        let mut candidates: HashSet<_> = known.keys().cloned().collect();
        candidates.extend(
            engine
                .skipped_versions
                .lock()
                .map_err(|_| Error::Config("skip outcomes lock poisoned".into()))?
                .keys()
                .cloned(),
        );
        let mut gone = Vec::new();
        match engine.provider.deletion_scope(&scope) {
            Ok(deletion_scope) => {
                for id in candidates {
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
            Ok(gone.iter().filter(|id| known.contains_key(*id)).count())
        })?;

        if !gone.is_empty() {
            let gone: HashSet<_> = gone.iter().collect();
            engine
                .skipped_versions
                .lock()
                .map_err(|_| Error::Config("skip outcomes lock poisoned".into()))?
                .retain(|resource, _| !gone.contains(resource));
            engine
                .pdf_diagnostics
                .lock()
                .map_err(|_| Error::Config("PDF diagnostics lock poisoned".into()))?
                .retain(|(resource, _), _| !gone.contains(resource));
        }
        sync.skipped_counts = skipped_counts(&sync.skipped);
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

    #[test]
    fn skipped_versions_stay_bounded_after_delete_cycles() {
        // Based on the review allocator probe: new identities in every cycle,
        // including skipped-only resources that never had cache bindings.
        let root = tempfile::tempdir().unwrap();
        let engine = SearchEngine::lexical().unwrap();
        for batch in 0..3 {
            for i in 0..1000 {
                std::fs::write(root.path().join(format!("bad-{batch}-{i}.txt")), [0xff]).unwrap();
            }
            let report = engine.index().paths([root.path()]).execute().unwrap();
            assert_eq!(report.sync.skipped.len(), 1000);
            assert_eq!(engine.skipped_versions.lock().unwrap().len(), 1000);
            for i in 0..1000 {
                std::fs::remove_file(root.path().join(format!("bad-{batch}-{i}.txt"))).unwrap();
            }
            let report = engine.index().paths([root.path()]).execute().unwrap();
            assert_eq!(report.sync.resources_discovered, 0);
            assert_eq!(report.resources_removed, 0, "no cached resources existed");
            assert!(engine.skipped_versions.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn skipped_versions_prune_only_confirmed_deletions_in_scope() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let deleted = first.path().join("deleted.txt");
        let excluded = first.path().join("excluded.txt");
        let outside = second.path().join("outside.txt");
        for path in [&deleted, &excluded, &outside] {
            std::fs::write(path, [0xff]).unwrap();
        }
        let engine = SearchEngine::lexical().unwrap();
        engine
            .index()
            .paths([first.path(), second.path()])
            .execute()
            .unwrap();
        let excluded_id = ResourceId::from_path(excluded.canonicalize().unwrap()).unwrap();
        let outside_id = ResourceId::from_path(outside.canonicalize().unwrap()).unwrap();
        std::fs::remove_file(deleted).unwrap();
        std::fs::remove_file(outside).unwrap();
        let report = engine
            .index()
            .paths([first.path()])
            .discovery(DiscoveryOptions {
                exclude: vec!["excluded.txt".into()],
                ..Default::default()
            })
            .execute()
            .unwrap();
        assert_eq!(report.sync.resources_discovered, 0);
        {
            let skipped = engine.skipped_versions.lock().unwrap();
            assert_eq!(skipped.len(), 2);
            assert!(
                skipped.contains_key(&excluded_id),
                "excluded is not deleted"
            );
            assert!(
                skipped.contains_key(&outside_id),
                "other scopes must survive"
            );
        }
        engine.index().paths([second.path()]).execute().unwrap();
        let skipped = engine.skipped_versions.lock().unwrap();
        assert_eq!(skipped.len(), 1);
        assert!(skipped.contains_key(&excluded_id));
    }

    fn worker_engine() -> SearchEngine {
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
                panic!("worker must not embed")
            }
        }
        SearchEngine::builder()
            .embedder(NoInference(crate::ModelIdentity {
                model_id: "test/worker".into(),
                revision: "1".into(),
                variant: "test".into(),
                dimension: crate::Dimension::D128,
                normalization: "l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            }))
            .build()
            .unwrap()
    }

    // A waiting notification is also a barrier: the previous iteration has dropped
    // its strong references and guards. No wall-clock sleeps are needed in tests.
    fn worker_ticks() -> (
        std::sync::mpsc::Sender<()>,
        std::sync::mpsc::Receiver<()>,
        impl FnMut() -> bool + Send,
    ) {
        let (ticks, receive) = std::sync::mpsc::channel();
        let (waiting, waits) = std::sync::mpsc::channel();
        (ticks, waits, move || {
            waiting.send(()).unwrap();
            receive.recv().is_ok()
        })
    }

    fn worker_wait(waits: &std::sync::mpsc::Receiver<()>) {
        waits
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_skip_codes_come_from_variants_not_display_text() {
        let resource = "mem://typed".parse().unwrap();
        assert_eq!(
            pdf_skip(
                &resource,
                Error::ParserPanicked {
                    resource: resource.clone(),
                    reason: "arbitrary sanitized reason"
                }
            )
            .code,
            SkipCode::PdfParserPanicked
        );
        assert_eq!(
            pdf_skip(
                &resource,
                Error::PdfPageDecode {
                    resource: resource.clone(),
                    reason: "arbitrary sanitized reason"
                }
            )
            .code,
            SkipCode::PdfParseFailed
        );
        assert_eq!(
            skip(&resource, "PDF parser panicked: forged string").code,
            SkipCode::Other
        );
    }

    #[test]
    fn cache_worker_duplicate_start_runs_only_one_worker() {
        use std::sync::atomic::AtomicUsize;
        let engine = worker_engine();
        let (ticks, waits, clock) = worker_ticks();
        let runs = Arc::new(AtomicUsize::new(0));
        let observed = runs.clone();
        let worker = engine
            .start_cache_scrubber_with(clock, move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        worker_wait(&waits);
        assert!(
            engine
                .start_cache_scrubber_with(
                    || panic!("duplicate worker started"),
                    |_| panic!("duplicate scrub")
                )
                .is_none()
        );
        engine.start_cache_scrubber();
        ticks.send(()).unwrap();
        worker_wait(&waits);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        drop(ticks);
        worker.join().unwrap();
    }

    #[test]
    fn cache_worker_excludes_active_indexing() {
        use std::sync::atomic::AtomicUsize;
        let engine = worker_engine();
        let (ticks, waits, clock) = worker_ticks();
        let runs = Arc::new(AtomicUsize::new(0));
        let observed = runs.clone();
        let worker = engine
            .start_cache_scrubber_with(clock, move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
            })
            .unwrap();
        worker_wait(&waits);
        let indexing = engine.indexing.lock().unwrap();
        ticks.send(()).unwrap();
        worker_wait(&waits);
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        drop(indexing);
        ticks.send(()).unwrap();
        worker_wait(&waits);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        drop(ticks);
        worker.join().unwrap();
    }

    fn join_worker_promptly(worker: std::thread::JoinHandle<()>) {
        let (completed, completion) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = completed.send(worker.join());
        });
        completion
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("worker did not exit within two seconds")
            .unwrap();
    }

    #[test]
    fn cache_worker_drop_interrupts_real_production_deadline_wait() {
        let engine = worker_engine();
        let weak_cache = Arc::downgrade(&engine.cache);
        let weak_indexing = Arc::downgrade(&engine.indexing);
        let stop = engine.scrubber_stop.clone();
        let worker = engine.spawn_cache_scrubber().unwrap();
        // The same production waiter will sleep for 60 seconds. Observing its
        // state under the mutex proves it entered the condvar wait; no tick or
        // wake is injected by this test. Only engine Drop may interrupt it.
        assert_eq!(
            MemoryCache::new().scrub_delay(std::time::Duration::from_secs(60)),
            std::time::Duration::from_secs(60)
        );
        let state = stop.state.lock().unwrap();
        let (state, _) = stop
            .wake
            .wait_timeout_while(state, std::time::Duration::from_secs(5), |state| {
                !state.waiting
            })
            .unwrap();
        assert!(
            state.waiting,
            "production worker did not enter its deadline wait"
        );
        assert!(!state.stopped);
        drop(state);
        drop(engine);
        assert!(weak_cache.upgrade().is_none());
        assert!(weak_indexing.upgrade().is_none());
        join_worker_promptly(worker);
    }

    #[test]
    fn cache_worker_drop_while_active_finishes_only_current_iteration() {
        let engine = worker_engine();
        let weak_cache = Arc::downgrade(&engine.cache);
        let weak_indexing = Arc::downgrade(&engine.indexing);
        let (ticks, waits, clock) = worker_ticks();
        let (entered, active) = std::sync::mpsc::channel();
        let (finish, release) = std::sync::mpsc::channel();
        let worker = engine
            .start_cache_scrubber_with(clock, move |_| {
                entered.send(()).unwrap();
                release
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            })
            .unwrap();
        worker_wait(&waits);
        ticks.send(()).unwrap();
        worker_wait(&active);
        drop(engine);
        assert!(
            weak_cache.upgrade().is_some(),
            "in-flight iteration retains its cache"
        );
        assert!(weak_indexing.upgrade().is_some());
        finish.send(()).unwrap();
        join_worker_promptly(worker);
        assert!(
            weak_cache.upgrade().is_none(),
            "released after current iteration"
        );
        assert!(weak_indexing.upgrade().is_none());
        assert!(
            waits.try_recv().is_err(),
            "must not start another wait after drop"
        );
    }

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
            "Example Person",
            "INC-18372",
            "where is Session::restore() called",
            "invoice_38192",
            "pricing page",
        ] {
            assert_eq!(w(q), 1.0, "{q}");
        }
        for q in [
            "how can I make tomato soup for dinner",
            "how to grow vegetables in a small garden",
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
    fn hybrid_rrf_k5_preserves_a_strong_semantic_lead() {
        assert_eq!(RRF_K, 5.0);
        let semantic: Vec<_> = (0..10)
            .map(|i| {
                let key = (format!("mem://rrf/{i}").parse().unwrap(), 0);
                (key, 1.0 - i as f32 * 0.05, 0)
            })
            .collect();
        // At k=60, the tenth semantic hit's lexical vote overtakes rank one.
        // At k=5, a prose keyword match still contributes without erasing that lead.
        let lexical = vec![(semantic[9].0.clone(), 1.0)];
        let hits = rank(SearchMode::Hybrid, &semantic, &lexical, 0.3, &[], 10);
        assert_eq!(hits[0].0, semantic[0].0);
        assert_eq!(hits[3].0, semantic[9].0);
        assert!((hits[0].1 - 10.0 / 13.0).abs() < 1e-6);
        assert!((hits[3].1 - 7.0 / 13.0).abs() < 1e-6);
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
        // Content ids are canonical; macOS temp directories sit behind /var.
        let root = dir.path().canonicalize().unwrap();
        let fixture = |name: &str, texts: &[&str]| {
            let path = root.join(name);
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
        let path = dir.path().canonicalize().unwrap().join("broken.pdf");
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
