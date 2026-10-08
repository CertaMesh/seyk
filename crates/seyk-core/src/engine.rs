//! The search engine: discovery, segmentation, lazy embedding, ranking.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
use crate::{Error, ResourceId, ResourceLocator, ResourceMetadata, Result};
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

pub type ProgressFn<'a> = &'a (dyn Fn(ProgressEvent) + Sync);

pub struct SearchEngineBuilder {
    provider: Arc<dyn ResourceProvider>,
    extractors: Vec<Arc<dyn Extractor>>,
    embedder: Option<Arc<dyn Embedder>>,
    cache: Option<Box<dyn EmbeddingCache>>,
    segmenter: TextSegmenter,
    batch_size: usize,
}

impl SearchEngineBuilder {
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

    pub fn build(mut self) -> Result<SearchEngine> {
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
            embedder,
            cache: Mutex::new(cache),
            segmenter: self.segmenter,
            profile,
            profile_id,
            batch_size: self.batch_size,
        })
    }
}

/// Search over provider resources; local files are the default.
pub struct SearchEngine {
    provider: Arc<dyn ResourceProvider>,
    extractors: Vec<Arc<dyn Extractor>>,
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
            provider: Arc::new(FsProvider),
            extractors: Vec::new(),
            embedder: None,
            cache: None,
            segmenter: TextSegmenter::default(),
            batch_size: 16,
        }
    }

    /// Extract readable evidence from a hit, verifying its full-resource hash
    /// before any extractor receives bytes. No indexing or model calls occur.
    pub fn extract(&self, hit: &SearchHit) -> Result<ExtractedContent> {
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
                    format!("{modality} search is not supported by this embedder"),
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
                    ])
                }
                #[cfg(not(feature = "pdf"))]
                {
                    r.fingerprint()
                }
            } else {
                r.fingerprint()
            };
            let cached = known.get(&r.id);
            // Listing already supplies a freshness token; only stat again before a read.
            if cached.is_some_and(|m| {
                m.version.matches(&r.version)
                    && m.descriptor_hash.as_ref() == Some(&descriptor_hash)
            }) {
                stats.resources_unchanged += 1;
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
                media_type: (r.kind == ResourceKind::Pdf).then(|| "application/pdf".into()),
                modality,
                size_bytes: bytes.len() as u64,
                modified_ns: r.modified_ns,
                content_hash,
            };
            let prepared = if r.kind == ResourceKind::Pdf {
                #[cfg(feature = "pdf")]
                {
                    self.prepare_pdf(r, &bytes).map(|(jobs, diagnostics)| {
                        stats
                            .skipped
                            .extend(diagnostics.into_iter().map(|reason| skip(&r.id, reason)));
                        jobs
                    })
                }
                #[cfg(not(feature = "pdf"))]
                {
                    Err("PDF feature is disabled".into())
                }
            } else {
                self.prepare(r, bytes)
            };
            match prepared {
                Ok(jobs) => pending.push(Pending {
                    resource: r.id.clone(),
                    metadata,
                    version,
                    descriptor_hash,
                    jobs,
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
        }
        Ok(upto.max(from))
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
        match self.embedder.embed_batch(&inputs) {
            Ok(vs) if vs.len() == inputs.len() => {
                vs.into_iter().map(|v| Ok(v.into_vec())).collect()
            }
            _ if inputs.len() > 1 => inputs
                .iter()
                .map(|i| {
                    self.embedder
                        .embed_batch(std::slice::from_ref(i))
                        .and_then(|mut v| {
                            v.pop().ok_or_else(|| {
                                seyk_embed::Error::Embedding("no vector returned".into())
                            })
                        })
                        .map(|e| e.into_vec())
                        .map_err(Error::from)
                })
                .collect(),
            Ok(_) => vec![Err(Error::Embedding(
                "embedder returned the wrong number of vectors".into(),
            ))],
            Err(e) => vec![Err(e.into())],
        }
    }

    fn supports_kind(&self, kind: ResourceKind) -> bool {
        match kind {
            ResourceKind::Pdf => {
                cfg!(feature = "pdf") && self.embedder.supports(crate::Modality::Text)
            }
            _ => self.embedder.supports(kind.modality()),
        }
    }

    #[cfg(feature = "pdf")]
    fn prepare_pdf(
        &self,
        r: &ProvidedResource,
        bytes: &[u8],
    ) -> std::result::Result<(Vec<Job>, Vec<String>), String> {
        let pages = crate::pdf::read_pages(bytes, &r.id).map_err(|e| e.to_string())?;
        let mut jobs = Vec::new();
        let mut diagnostics = Vec::new();
        for page in pages {
            if !crate::pdf::usable(&page.text) {
                diagnostics.push(format!(
                    "PDF page {} has no usable embedded text; scan/OCR support is not enabled",
                    page.page
                ));
                continue;
            }
            let digest = blake3::hash(page.text.as_bytes()).to_hex().to_string();
            for segment in self.segmenter.segment(&page.text, TextFormat::Plain) {
                let title = segment_title(r.title.as_deref(), None);
                jobs.push(Job {
                    segment: CachedSegment {
                        locator: ResourceLocator::PageText {
                            page: page.page,
                            byte_start: segment.byte_range.start as u64,
                            byte_end: segment.byte_range.end as u64,
                            text_version: crate::pdf::TEXT_VERSION.into(),
                            page_text_hash: digest.clone(),
                        },
                        heading: None,
                        input_hash: input_hash(&[
                            "text",
                            title.as_deref().unwrap_or(""),
                            &segment.text,
                        ]),
                    },
                    lexical_body: segment.text.clone(),
                    input: JobInput::Text {
                        title,
                        text: segment.text,
                    },
                });
            }
        }
        if jobs.is_empty() {
            return Err("PDF has no usable embedded text; scan/OCR support is not enabled".into());
        }
        Ok((jobs, diagnostics))
    }

    /// Turn a resource into embedding jobs. Errors are reasons to skip the resource.
    fn prepare(
        &self,
        r: &ProvidedResource,
        bytes: Vec<u8>,
    ) -> std::result::Result<Vec<Job>, String> {
        match r.kind {
            ResourceKind::Pdf => {
                #[cfg(feature = "pdf")]
                {
                    self.prepare_pdf(r, &bytes).map(|(jobs, _)| jobs)
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
                Ok(self
                    .segmenter
                    .segment(&text, format)
                    .into_iter()
                    .map(|s| {
                        let title = segment_title(stem.as_deref(), s.heading.as_deref());
                        let model_text = match format {
                            TextFormat::Markdown => strip_link_targets(&s.text),
                            TextFormat::Plain | TextFormat::Code => s.text.clone(),
                        };
                        let hash =
                            input_hash(&["text", title.as_deref().unwrap_or(""), &model_text]);
                        Job {
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
                            // Index the original source, including link targets.
                            lexical_body: s.text,
                            input: JobInput::Text {
                                title,
                                text: model_text,
                            },
                        }
                    })
                    .collect())
            }
            ResourceKind::Image => {
                let hash = input_hash(&["image", &blake3::hash(&bytes).to_hex()]);
                Ok(vec![Job {
                    segment: CachedSegment {
                        locator: ResourceLocator::WholeResource,
                        heading: None,
                        input_hash: hash,
                    },
                    // Images are indexed by name and path only.
                    lexical_body: String::new(),
                    input: JobInput::Image { bytes },
                }])
            }
        }
    }
}

struct Pending {
    descriptor_hash: String,
    resource: ResourceId,
    version: ResourceVersion,
    metadata: ResourceMetadata,
    jobs: Vec<Job>,
}

struct Job {
    /// `segment.input_hash` identifies the model input; identical inputs in
    /// different resources are embedded once.
    segment: CachedSegment,
    lexical_body: String,
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
        let scope = ResourceScope::from_paths(&self.paths, self.roots, self.discovery)?;
        let resources = engine.provider.list(&scope)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let sync = if self.sync {
            engine.sync(&resources, self.progress)?
        } else {
            SyncStats {
                resources_discovered: resources.len(),
                ..Default::default()
            }
        };

        let pid = engine.profile_id.as_str();
        let wanted: HashSet<ResourceId> = resources
            .iter()
            .filter(|r| engine.supports_kind(r.kind))
            .map(|r| r.id.clone())
            .collect();
        let metadata = engine.with_cache(|c| c.resource_index(pid))?;
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
            let mut index = BruteForceIndex::with_capacity(dim, stored.len());
            let mut keys = Vec::with_capacity(stored.len());
            for v in stored {
                // A wrong-sized vector means a corrupt cache; never compare it.
                if v.vector.len() == dim {
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
            let query = LexicalQuery::new(self.queries.iter().map(String::as_str));
            lexical = engine
                .with_cache(|c| c.lexical_search(pid, &query, &wanted, depth))?
                .into_iter()
                .map(|h| ((h.resource, h.ordinal), h.score))
                .collect();
            named = resources
                .iter()
                .filter(|r| {
                    self.queries
                        .iter()
                        .any(|q| names_file(q, Path::new(&r.id.labels().1)))
                })
                .map(|r| (r.id.clone(), 0))
                .collect();
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
    segments: HashMap<SegmentKey, Option<CachedSegment>>,
}

impl SegmentTable {
    fn insert(&mut self, path: ResourceId, ordinal: usize, segment: CachedSegment) -> SegmentKey {
        let key = (path, ordinal);
        self.segments.insert(key.clone(), Some(segment));
        key
    }

    fn get_or_load(
        &mut self,
        engine: &SearchEngine,
        pid: &str,
        key: &SegmentKey,
    ) -> Result<Option<&CachedSegment>> {
        if !self.segments.contains_key(key) {
            let segment = engine
                .with_cache(|c| c.resource(pid, &key.0))?
                .and_then(|r| r.segments.into_iter().nth(key.1));
            self.segments.insert(key.clone(), segment);
        }
        Ok(self.segments[key].as_ref())
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
        let scope = ResourceScope::from_paths(&self.paths, self.roots, self.discovery)?;
        let resources = engine.provider.list(&scope)?;
        if let Some(p) = self.progress {
            p(ProgressEvent::Discovered {
                resources: resources.len(),
            });
        }
        let mut sync = engine.sync(&resources, self.progress)?;

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
                    let text = crate::pdf::read_page(bytes, path, *page).ok();
                    if let Some(t) = &text {
                        if self.page_bytes + t.len() > crate::pdf::MAX_DOCUMENT_TEXT_BYTES {
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
            let jobs = engine.prepare(&resource, bytes.to_vec()).unwrap();
            assert_eq!(jobs.len(), 1);
            assert_eq!(
                jobs[0].segment.input_hash,
                golden["input_hashes"][name].as_str().unwrap(),
                "{name}"
            );
        }
    }
}
