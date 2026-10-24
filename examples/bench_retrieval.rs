//! Public retrieval harness. See docs/benchmarks.md. Never uses the default cache.
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use seyk::discovery::{DiscoveryOptions, PolicyRoot, ResourceKind, discover};
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
use seyk::{Dimension, ResourceLocator, SearchEngine, SearchHit, SearchMode, SqliteCache};
use seyk::{ResourceId, SkipCode, SyncStats};

const PDF_PROOF: &str = "pdf-policy-miss-v2";
// Immutable cache ownership protocol; independent of evaluation acceptance.
const PDF_OWNER_PROOF: &str = "pdf-policy-miss-v1";
const PDF_LIMIT: u64 = 33_554_432;
const NO_TEXT: &str = if cfg!(feature = "pdf-render") {
    "no usable embedded text; image embedding is disabled"
} else {
    "no usable embedded text; PDF rendering is not compiled in (feature `pdf-render`)"
};

#[derive(Parser)]
struct Args {
    #[arg(long, value_parser = ["nanoscifact", "scifact", "nfcorpus", "irpapers", "irpapers-pdf"])]
    ds: String,
    #[arg(long, default_value_t = 256)]
    dimension: usize,
    #[arg(long, default_value = "q8")]
    precision: String,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value = "hybrid,semantic,lexical", value_delimiter = ',', value_parser = ["hybrid", "semantic", "lexical"])]
    modes: Vec<String>,
    #[arg(long, default_value_t = 200)]
    depth: usize,
    /// Check rankings without recording performance measurements.
    #[arg(long)]
    rank_only: bool,
    /// Isolate a non-PDF regression cache under cache/<dataset>/tasks/<session>.
    #[arg(long)]
    cache_session: Option<String>,
    /// Diagnostic only; never changes the engine default.
    #[arg(long)]
    lexical_weight: Option<f32>,
    /// Separate warm reruns without overwriting their reference run.
    #[arg(long, default_value = "run")]
    label: String,
    /// New task-owned PDF cache; use a new opaque session for each task.
    #[arg(long)]
    pdf_cache_session: Option<String>,
    /// Resume only the matching task ownership receipt (never a shared cache).
    #[arg(long)]
    resume_pdf_cache: bool,
}

fn opaque_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 96
        && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Exclusion {
    key: String,
    blake3: String,
    size_bytes: u64,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct OmittedPage {
    key: String,
    page: u32,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PdfCoverage {
    proof: &'static str,
    limit: u64,
    full_resources: usize,
    eligible_keys: BTreeSet<String>,
    exclusions: Vec<Exclusion>,
    omitted_text_pages: Vec<OmittedPage>,
}

/// Length and digest are observed together, once per validation boundary.
/// Discovery remains the authority for the exact set admitted by the default.
fn pdf_inventory(
    root: &Path,
    corpus: &Path,
    manifest: &[u8],
    keys: &HashSet<String>,
    options: &DiscoveryOptions,
) -> Result<(PdfCoverage, HashMap<ResourceId, String>)> {
    let policy = PolicyRoot::new(corpus)?;
    ensure!(
        options.max_file_bytes == PDF_LIMIT,
        "PDF default policy changed"
    );
    let provenance: serde_json::Value = serde_json::from_slice(manifest)?;
    ensure!(
        fs::read(confined(root, "meta/irpapers-pdf/manifest.json")?)? == manifest,
        "manifest changed during run"
    );
    let metadata = provenance["metadata_blake3"]
        .as_object()
        .context("metadata hashes")?;
    ensure!(
        metadata.keys().map(String::as_str).collect::<BTreeSet<_>>()
            == BTreeSet::from(["docmap.tsv", "queries.jsonl", "qrels.tsv"]),
        "metadata identity mismatch"
    );
    for (name, digest) in metadata {
        let bytes = fs::read(confined(root, &format!("meta/irpapers-pdf/{name}"))?)?;
        ensure!(
            digest.as_str() == Some(blake3::hash(&bytes).to_hex().as_str()),
            "metadata changed"
        );
    }
    let files = provenance["files"].as_object().context("manifest files")?;
    ensure!(
        files.keys().cloned().collect::<BTreeSet<_>>()
            == keys.iter().map(|k| format!("{k}.pdf")).collect(),
        "manifest/docmap mismatch"
    );
    let mut seen = HashSet::new();
    let mut eligible = HashMap::new();
    let mut expected = BTreeMap::new();
    let mut exclusions = Vec::new();
    for entry in fs::read_dir(corpus)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "nonregular PDF corpus entry");
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("bad PDF name"))?;
        let key = name
            .strip_suffix(".pdf")
            .context("unexpected PDF extension")?;
        ensure!(
            keys.contains(key) && seen.insert(key.to_owned()),
            "unknown/duplicate PDF identity"
        );
        let path = entry.path();
        seyk::discovery::ensure_allowed_path(&path)?;
        let bytes = fs::read(&path)?;
        let size = u64::try_from(bytes.len())?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        ensure!(files[&name].as_str() == Some(&digest), "PDF hash changed");
        if size > options.max_file_bytes {
            exclusions.push(Exclusion {
                key: key.into(),
                blake3: digest,
                size_bytes: size,
                reason: "encoded_size_exceeds_default".into(),
            });
        } else {
            let canonical = path.canonicalize()?;
            eligible.insert(ResourceId::from_path(&canonical)?, key.into());
            expected.insert(canonical, size);
        }
    }
    ensure!(&seen == keys, "incomplete PDF inventory");
    let found = discover(&[corpus.to_path_buf()], options)?;
    ensure!(
        found.len() == expected.len()
            && found.iter().all(
                |r| r.kind == ResourceKind::Pdf && expected.get(&r.path) == Some(&r.size_bytes)
            ),
        "default discovery identity/size mismatch"
    );
    ensure!(
        policy.revalidate().is_ok_and(|v| v),
        "PDF root changed or denied"
    );
    exclusions.sort_by(|a, b| a.key.cmp(&b.key));
    Ok((
        PdfCoverage {
            proof: PDF_PROOF,
            limit: PDF_LIMIT,
            full_resources: keys.len(),
            eligible_keys: eligible.values().cloned().collect(),
            exclusions,
            omitted_text_pages: Vec::new(),
        },
        eligible,
    ))
}

fn pdf_diagnostics(
    sync: &SyncStats,
    eligible: &HashMap<ResourceId, String>,
    fresh: bool,
) -> Result<Vec<OmittedPage>> {
    ensure!(
        sync.resources_discovered == eligible.len()
            && sync.resources_updated == if fresh { eligible.len() } else { 0 }
            && sync.resources_unchanged == if fresh { 0 } else { eligible.len() }
            && sync.skipped.is_empty()
            && sync.skipped_counts.is_empty(),
        "incomplete PDF preparation/sync"
    );
    let mut pages = BTreeSet::new();
    let mut identities = HashSet::new();
    for page in &sync.page_diagnostics {
        let key = eligible
            .get(&page.resource)
            .context("ineligible diagnostic")?;
        ensure!(
            page.page > 0 && identities.insert((key, page.page)),
            "fatal PDF page diagnostic"
        );
        ensure!(
            pages.insert(OmittedPage {
                key: key.clone(),
                page: page.page,
                reason: pdf_page_reason(page)?.into()
            }),
            "duplicate page diagnostic"
        );
    }
    Ok(pages.into_iter().collect())
}

fn pdf_page_reason(page: &seyk::PageDiagnostic) -> Result<&'static str> {
    match page.code {
        SkipCode::PdfParseFailed => Ok("pdf_parse_failed"),
        SkipCode::PdfParserPanicked => Ok("pdf_parser_panicked"),
        SkipCode::Other if page.reason == NO_TEXT => Ok("no_usable_embedded_text"),
        SkipCode::Other => bail!("fatal PDF page diagnostic"),
    }
}

// Only serialize fixed parser codes, never dependency error/panic payloads.
fn pdf_safe_sync(sync: &SyncStats) -> Result<SyncStats> {
    let mut safe = sync.clone();
    for page in &mut safe.page_diagnostics {
        if page.code != SkipCode::Other {
            page.reason = pdf_page_reason(page)?.into();
        }
    }
    Ok(safe)
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct PdfOwner {
    purpose: String,
    version: u32,
    session: String,
    dataset: String,
    manifest_blake3: String,
    profile_id: String,
    proof: String,
}

impl PdfOwner {
    fn new(session: &str, manifest: &[u8], profile_id: &str) -> Self {
        Self {
            purpose: "seyk-benchmark-task".into(),
            version: 1,
            session: session.into(),
            dataset: "irpapers-pdf".into(),
            manifest_blake3: blake3::hash(manifest).to_hex().to_string(),
            profile_id: profile_id.into(),
            proof: PDF_OWNER_PROOF.into(),
        }
    }
}

/// The descriptor owns the advisory lock until the engine and all modes finish.
struct PdfSession {
    _lock: File,
    cache: PathBuf,
}

impl PdfSession {
    #[cfg(not(unix))]
    fn open(_root: &Path, _owner: &PdfOwner, _resume: bool) -> Result<Self> {
        bail!("PDF task cache currently requires Unix private permissions and locking")
    }

    #[cfg(unix)]
    fn open(root: &Path, owner: &PdfOwner, resume: bool) -> Result<Self> {
        use std::fs::OpenOptions;
        ensure!(opaque_label(&owner.session), "invalid PDF cache session");
        let relative = format!("cache/irpapers-pdf/tasks/{}", owner.session);
        let dir = confined(root, &relative)?;
        if resume {
            ensure!(dir.is_dir(), "missing task directory");
        } else {
            fs::create_dir_all(dir.parent().context("task parent")?)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                fs::DirBuilder::new().mode(0o700).create(&dir)?;
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let meta = fs::metadata(&dir)?;
            ensure!(
                meta.permissions().mode() & 0o077 == 0 && meta.uid() == unsafe { libc::geteuid() },
                "task directory must be private and owned by this user"
            );
        }
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid task entry"))?;
            ensure!(
                [
                    "owner.json",
                    "task.lock",
                    "cache.sqlite",
                    "cache.sqlite-wal",
                    "cache.sqlite-shm",
                    "cache.sqlite-journal"
                ]
                .contains(&name.as_str())
                    && entry.file_type()?.is_file(),
                "unexpected task cache entry"
            );
        }
        let lock_path = confined(root, &format!("{relative}/task.lock"))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(!resume)
            .open(lock_path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: a live descriptor; flock is released by closing it or process death.
            ensure!(
                unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
                "PDF task cache has a conflicting writer"
            );
        }
        let receipt = confined(root, &format!("{relative}/owner.json"))?;
        if resume {
            ensure!(
                serde_json::from_slice::<PdfOwner>(&fs::read(receipt)?)? == *owner,
                "PDF ownership receipt mismatch"
            );
        } else {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(receipt)?;
            serde_json::to_writer(&mut file, owner)?;
            file.sync_all()?;
        }
        let cache = confined(root, &format!("{relative}/cache.sqlite"))?;
        if resume {
            ensure!(cache.is_file(), "missing task cache");
        }
        Ok(Self { _lock: lock, cache })
    }
}

fn reset_pdf_bindings(engine: &SearchEngine, eligible: &HashMap<ResourceId, String>) -> Result<()> {
    engine.with_cache(|cache| {
        let profile = engine.profile_id();
        if cache
            .stats()?
            .profiles
            .iter()
            .any(|p| p.profile_id != profile)
            || cache
                .resource_index(profile)?
                .keys()
                .any(|id| !eligible.contains_key(id))
        {
            return Err(seyk::Error::Config(
                "unexpected PDF task profile/resource".into(),
            ));
        }
        cache.remove_resources(profile, &eligible.keys().cloned().collect::<Vec<_>>())?;
        if !cache.resource_index(profile)?.is_empty() {
            return Err(seyk::Error::Config("PDF bindings survived reset".into()));
        }
        Ok(())
    })?;
    Ok(())
}

#[derive(Deserialize)]
struct Query {
    qid: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct Ranked {
    key: String,
    rank: usize,
    score: f32,
    semantic: Option<f32>,
    lexical: Option<f32>,
    filename: bool,
    locator: ResourceLocator,
}

fn pdf_ranked(hit: &Ranked, omitted: &[OmittedPage]) -> Result<()> {
    let page = match hit.locator {
        ResourceLocator::PageText { page, .. } | ResourceLocator::Page { page } => page,
        _ => bail!("PDF hit lacks physical page"),
    };
    ensure!(
        page > 0 && !omitted.iter().any(|p| p.key == hit.key && p.page == page),
        "omitted/failed PDF page hit"
    );
    Ok(())
}

/// Preserve engine order and the best segment's signals. Do not re-sort scores:
/// hybrid scores can tie, and their engine order is part of what we evaluate.
fn collapse(hits: &[Ranked], pages: bool) -> Vec<Ranked> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for hit in hits {
        let page = if pages {
            match hit.locator {
                ResourceLocator::PageText { page, .. } => Some(page),
                _ => continue,
            }
        } else {
            None
        };
        if seen.insert((&hit.key, page)) {
            let mut hit = hit.clone();
            hit.rank = out.len() + 1;
            out.push(hit);
        }
    }
    out
}

fn ranked(hit: SearchHit, corpus: &Path, keys: &HashSet<String>) -> Result<Ranked> {
    let path = hit.resource.to_file_path().context("non-file hit")?;
    ensure!(
        seyk::ResourceId::from_path(path.parent().context("missing hit parent")?)?
            == seyk::ResourceId::from_path(corpus)?,
        "hit outside benchmark corpus"
    );
    let key = path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("bad key")?;
    ensure!(keys.contains(key), "hit missing from docmap");
    Ok(Ranked {
        key: key.to_owned(),
        rank: 0,
        score: hit.score,
        semantic: hit.signals.semantic,
        lexical: hit.signals.lexical,
        filename: hit.signals.filename,
        locator: hit.locator,
    })
}

#[cfg(unix)]
fn load_average() -> Option<[f64; 3]> {
    let mut loads = [0.0; 3];
    // SAFETY: loads has space for the three requested values.
    (unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) } == 3).then_some(loads)
}
#[cfg(not(unix))]
fn load_average() -> Option<[f64; 3]> {
    None
}

fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(output.status.success(), "{program} failed");
    Ok(String::from_utf8(output.stdout)?.trim().into())
}

/// Reject symlink escapes, including any existing ancestor of an output path.
fn confined(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = root.join(relative);
    for ancestor in path.ancestors().take_while(|p| *p != root) {
        if ancestor.symlink_metadata().is_ok() {
            ensure!(
                !ancestor.symlink_metadata()?.file_type().is_symlink(),
                "symlink in benchmark path"
            );
        }
    }
    Ok(path)
}

fn write_row(writer: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writeln!(writer)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let pdf = args.ds == "irpapers-pdf";
    ensure!(
        pdf || (args.pdf_cache_session.is_none() && !args.resume_pdf_cache),
        "PDF cache flags on text track"
    );
    ensure!(
        !pdf || args.pdf_cache_session.as_deref().is_some_and(opaque_label),
        "PDF track requires --pdf-cache-session"
    );
    ensure!(args.depth >= 100, "depth must be at least 100 segments");
    ensure!(
        !args.label.is_empty()
            && args
                .label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "invalid label"
    );
    if let Some(w) = args.lexical_weight {
        ensure!(w.is_finite() && w >= 0.0, "invalid weight");
    }
    let modes: Vec<SearchMode> = args
        .modes
        .iter()
        .map(|m| m.parse())
        .collect::<seyk::Result<_>>()?;
    ensure!(
        !modes.is_empty() && args.modes.iter().collect::<HashSet<_>>().len() == modes.len(),
        "empty or duplicate modes"
    );
    ensure!(
        !pdf || args
            .modes
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == BTreeSet::from(["hybrid", "semantic", "lexical"]),
        "PDF proof requires all three modes"
    );
    let root = dirs::home_dir()
        .context("no home directory")?
        .join(".cache/seyk-bench");
    ensure!(
        !root.symlink_metadata()?.file_type().is_symlink(),
        "benchmark root must not be a symlink"
    );
    let pdf_policy = if pdf {
        Some(PolicyRoot::new(&confined(&root, "corpus/irpapers-pdf")?)?)
    } else {
        None
    };
    let root = root.canonicalize()?;
    let corpus = confined(&root, &format!("corpus/{}", args.ds))?;
    let manifest = fs::read(confined(&root, &format!("meta/{}/manifest.json", args.ds))?)?;
    let provenance: serde_json::Value = serde_json::from_slice(&manifest)?;
    ensure!(
        provenance["dataset"] == args.ds,
        "wrong materialized dataset"
    );
    for (name, digest) in provenance["metadata_blake3"]
        .as_object()
        .context("missing metadata hashes")?
    {
        ensure!(
            ["docmap.tsv", "queries.jsonl", "qrels.tsv"].contains(&name.as_str()),
            "unexpected metadata file"
        );
        let bytes = fs::read(confined(&root, &format!("meta/{}/{name}", args.ds))?)?;
        ensure!(
            digest.as_str() == Some(blake3::hash(&bytes).to_hex().as_str()),
            "metadata changed since materialization"
        );
    }
    let docmap = fs::read_to_string(confined(&root, &format!("meta/{}/docmap.tsv", args.ds))?)?;
    let key_rows: Vec<String> = docmap
        .lines()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap_or("").to_owned())
        .collect();
    let keys: HashSet<String> = key_rows.iter().cloned().collect();
    ensure!(!keys.is_empty(), "empty docmap");
    if pdf {
        let mut ids = HashSet::new();
        ensure!(
            docmap.lines().next() == Some("key\tdocid") && key_rows.len() == keys.len(),
            "invalid/duplicate PDF docmap"
        );
        for line in docmap.lines().skip(1) {
            let (key, id) = line.split_once('\t').context("invalid docmap row")?;
            ensure!(
                id.parse::<u64>()
                    .is_ok_and(|v| v > 0 && v.to_string() == id)
                    && ids.insert(id)
                    && key == &blake3::hash(id.as_bytes()).to_hex()[..16],
                "invalid PDF identity"
            );
        }
        // Read the pinned count contract from the existing source of truth.
        let section = include_str!("../scripts/bench/datasets.toml")
            .split("[irpapers-pdf]\n")
            .nth(1)
            .context("PDF config")?;
        for field in ["kind", "revision", "license", "documents", "queries"] {
            let value = section
                .lines()
                .take_while(|l| !l.starts_with('['))
                .find_map(|l| l.strip_prefix(&format!("{field} = ")))
                .context("PDF config field")?;
            let expected: serde_json::Value = serde_json::from_str(value)?;
            ensure!(
                provenance["source"][field] == expected,
                "PDF source config mismatch"
            );
        }
        let source_files = section
            .split("[[irpapers-pdf.files]]\n")
            .skip(1)
            .map(|part| {
                ["name", "url", "sha256"]
                    .into_iter()
                    .map(|field| {
                        let value = part
                            .lines()
                            .take_while(|l| !l.starts_with('['))
                            .find_map(|l| l.strip_prefix(&format!("{field} = ")))
                            .context("PDF source file field")?;
                        Ok((
                            field.to_owned(),
                            serde_json::from_str::<serde_json::Value>(value)?,
                        ))
                    })
                    .collect::<Result<serde_json::Map<_, _>>>()
                    .map(serde_json::Value::Object)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            provenance["source"]["files"] == json!(source_files),
            "PDF source file config mismatch"
        );
        ensure!(
            keys.len() as u64
                == provenance["source"]["documents"]
                    .as_u64()
                    .context("documents")?,
            "PDF document count"
        );
    }
    let discovery = DiscoveryOptions::default();
    ensure!(
        !pdf || cfg!(feature = "pdf"),
        "PDF track requires the pdf feature"
    );
    let extension = if pdf { "pdf" } else { "txt" };
    let mut count = 0;
    for entry in fs::read_dir(&corpus)? {
        if pdf {
            break;
        }
        let entry = entry?;
        ensure!(
            entry.file_type()?.is_file(),
            "corpus contains a non-regular file"
        );
        let path = entry.path();
        ensure!(
            path.extension().and_then(|s| s.to_str()) == Some(extension),
            "unexpected corpus extension"
        );
        ensure!(
            keys.contains(path.file_stem().and_then(|s| s.to_str()).unwrap_or("")),
            "unknown corpus key"
        );
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("bad file name")?;
        let digest = blake3::hash(&fs::read(&path)?).to_hex().to_string();
        ensure!(
            provenance["files"][name].as_str() == Some(&digest),
            "corpus changed since materialization"
        );
        count += 1;
    }
    if !pdf {
        ensure!(count == keys.len(), "incomplete corpus");
    }
    let mut pdf_proof = if pdf {
        Some(pdf_inventory(&root, &corpus, &manifest, &keys, &discovery)?)
    } else {
        None
    };
    if pdf {
        count = keys.len();
    }
    let queries: Vec<Query> =
        fs::read_to_string(confined(&root, &format!("meta/{}/queries.jsonl", args.ds))?)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !queries.is_empty()
            && queries.iter().map(|q| &q.qid).collect::<HashSet<_>>().len() == queries.len(),
        "empty or duplicate queries"
    );
    ensure!(
        !pdf || queries.len() as u64
            == provenance["source"]["queries"]
                .as_u64()
                .context("queries")?,
        "PDF query count"
    );
    if pdf {
        let qrels = fs::read_to_string(confined(&root, "meta/irpapers-pdf/qrels.tsv")?)?;
        let mut gold_queries = HashSet::new();
        let papers: HashSet<_> = docmap
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once('\t').map(|(_, id)| id))
            .collect();
        ensure!(
            qrels.lines().next() == Some("query-id\tcorpus-id\tscore"),
            "PDF qrels header"
        );
        for line in qrels.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            ensure!(
                fields.len() == 3 && fields[2] == "1" && gold_queries.insert(fields[0]),
                "invalid/duplicate PDF qrel"
            );
            let (paper, page) = fields[1].split_once('#').context("PDF page qrel")?;
            ensure!(
                papers.contains(paper)
                    && page
                        .parse::<u32>()
                        .is_ok_and(|n| n > 0 && n.to_string() == page),
                "PDF qrel target"
            );
        }
        ensure!(
            gold_queries == queries.iter().map(|q| q.qid.as_str()).collect(),
            "PDF original question/qrel mismatch"
        );
    }
    if let Some(session) = &args.cache_session {
        ensure!(
            !pdf && opaque_label(session),
            "cache-session requires a non-PDF dataset and an opaque label"
        );
    }
    let started = (!args.rank_only).then(Instant::now);
    let load_before = load_average();
    let mut config = GemmaConfig::default();
    config.dimension = Dimension::try_from(args.dimension)?;
    config.precision = args.precision.parse()?;
    config.threads = args.threads;
    config.vision = false;
    config.audio = false;
    let hf_home = confined(&root, "huggingface")?;
    ensure!(
        std::env::var_os("HF_HOME").map(PathBuf::from).as_ref() == Some(&hf_home),
        "set HF_HOME to ~/.cache/seyk-bench/huggingface; default model cache is not used"
    );
    let model_dir = seyk::embedder::gemma::locate(&config)?;
    for file in config.files() {
        ensure!(
            model_dir.join(file).canonicalize()?.starts_with(&root),
            "model file outside benchmark root"
        );
    }
    // Read-only local model resolution; this example never downloads weights.
    let embedder = Arc::new(GemmaEmbedder::load(config)?);
    let profile_id = SearchEngine::builder()
        .shared_embedder(embedder.clone())
        .build()?
        .profile_id()
        .to_owned();
    let pdf_session = args
        .pdf_cache_session
        .as_ref()
        .map(|session| {
            PdfSession::open(
                &root,
                &PdfOwner::new(session, &manifest, &profile_id),
                args.resume_pdf_cache,
            )
        })
        .transpose()?;
    let cache = if let Some(session) = &pdf_session {
        session.cache.clone()
    } else if let Some(session) = &args.cache_session {
        confined(
            &root,
            &format!("cache/{}/tasks/{session}/{profile_id}.sqlite", args.ds),
        )?
    } else {
        confined(&root, &format!("cache/{}/{profile_id}.sqlite", args.ds))?
    };
    fs::create_dir_all(cache.parent().context("cache parent")?)?;
    let engine = SearchEngine::builder()
        .shared_embedder(embedder)
        .cache(SqliteCache::open(cache)?)
        .build()?;
    let git_sha = command("git", &["rev-parse", "HEAD"])?;
    let dirty = !command("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty();
    let run_dir = confined(&root, &format!("runs/{}", args.ds))?;
    fs::create_dir_all(&run_dir)?;
    let weight = args
        .lexical_weight
        .map_or("default".to_owned(), |w| w.to_string());
    let run_id = format!("{git_sha}-{profile_id}-w{weight}-{}", args.label);
    for mode in &args.modes {
        ensure!(
            !run_dir.join(format!("{run_id}-{mode}.jsonl")).exists(),
            "run exists; use a new --label"
        );
    }
    let mut before_stats = engine.with_cache(|c| c.stats())?;
    if pdf {
        before_stats.location = None;
    }
    if let Some((_, eligible)) = &pdf_proof {
        reset_pdf_bindings(&engine, eligible)?;
    }
    eprintln!(
        "indexing {count} resources; embedding progress reports segments needed before inference"
    );
    let progress = |event| eprintln!("{event:?}");
    let index = engine
        .index()
        .paths([&corpus])
        .discovery(discovery.clone())
        .progress(&progress)
        .execute()?;
    // Report parser failures before completeness checks, without leaking corpus
    // names, source bytes or dependency panic payloads into benchmark logs.
    for skipped in &index.sync.skipped_counts {
        eprintln!(
            "skipped resources: count={} code={:?}",
            skipped.count, skipped.code
        );
    }
    let mut diagnostics = std::collections::BTreeMap::new();
    for page in &index.sync.page_diagnostics {
        if let Some((_, eligible)) = &pdf_proof {
            ensure!(
                eligible.contains_key(&page.resource),
                "ineligible PDF diagnostic"
            );
        }
        *diagnostics
            .entry((
                page.resource
                    .to_file_path()
                    .context("non-file diagnostic")?,
                page.code,
            ))
            .or_insert(0usize) += 1;
    }
    for ((path, code), pages) in diagnostics {
        let hash = blake3::hash(&fs::read(path)?);
        eprintln!("page diagnostic: hash={hash} pages={pages} code={code:?}");
    }
    ensure!(
        index.sync.resources_discovered == pdf_proof.as_ref().map_or(count, |(_, e)| e.len()),
        "discovery omitted benchmark resources"
    );
    if !index.sync.skipped.is_empty() {
        bail!(
            "{} resources skipped; refusing incomplete run",
            index.sync.skipped.len()
        );
    }
    if let Some((coverage, eligible)) = &mut pdf_proof {
        ensure!(
            pdf_policy
                .as_ref()
                .context("PDF policy root")?
                .revalidate()
                .is_ok_and(|v| v),
            "PDF root identity changed"
        );
        ensure!(index.resources_removed == 0, "PDF resources removed");
        coverage.omitted_text_pages = pdf_diagnostics(&index.sync, eligible, true)?;
        let (mut observed, ids) = pdf_inventory(&root, &corpus, &manifest, &keys, &discovery)?;
        observed.omitted_text_pages = coverage.omitted_text_pages.clone();
        ensure!(
            &observed == coverage && &ids == eligible,
            "PDF corpus changed during index"
        );
    }
    let mut after_stats = engine.with_cache(|c| c.stats())?;
    if pdf {
        after_stats.location = None;
    }
    let mut index_evidence = serde_json::to_value(&index)?;
    if pdf {
        index_evidence["sync"] = serde_json::to_value(pdf_safe_sync(&index.sync)?)?;
    }
    if args.rank_only {
        index_evidence
            .as_object_mut()
            .context("index evidence")?
            .remove("total_ms");
        index_evidence["sync"]
            .as_object_mut()
            .context("sync evidence")?
            .remove("embed_ms");
    }
    let env = json!({"type":"env", "schema":1, "dataset":args.ds, "dataset_manifest":provenance,
        "manifest_blake3":blake3::hash(&manifest).to_hex().to_string(),
        "git_sha":git_sha, "dirty":dirty, "run_id":run_id, "profile_id":profile_id,
        "profile":engine.profile(), "precision":args.precision, "dimension":args.dimension,
        "threads":args.threads, "depth":args.depth, "lexical_weight":args.lexical_weight,
        "rank_only":args.rank_only,"cache_session":args.cache_session,
        "os":std::env::consts::OS, "arch":std::env::consts::ARCH,
        "logical_cpus":std::thread::available_parallelism().ok().map(|v|v.get()),
        "rust":command("rustc", &["--version"] )?, "load_before":load_before,
        "load_after_index":load_average(), "index":index_evidence,
        "cache_before":before_stats, "cache_after":after_stats,
        "pdf_coverage":pdf_proof.as_ref().map(|(c,_)| c),
        "pdf_cache_session":args.pdf_cache_session});
    for (name, mode) in args.modes.iter().zip(modes) {
        let final_path = confined(&root, &format!("runs/{}/{run_id}-{name}.jsonl", args.ds))?;
        let partial = final_path.with_extension("partial");
        ensure!(
            !partial
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink()),
            "symlink output"
        );
        let mut writer = BufWriter::new(File::create(&partial)?);
        write_row(&mut writer, &env)?;
        for query in &queries {
            let start = (!args.rank_only).then(Instant::now);
            let mut request = engine
                .search(&query.text)
                .paths([&corpus])
                .discovery(discovery.clone())
                .mode(mode)
                .limit(args.depth)
                .previews(None);
            if let Some(w) = args.lexical_weight {
                request = request.lexical_weight(w);
            }
            let results = request.execute()?;
            if let Some((coverage, eligible)) = &pdf_proof {
                ensure!(
                    pdf_diagnostics(&results.stats.sync, eligible, false)?
                        == coverage.omitted_text_pages,
                    "PDF diagnostics changed during queries"
                );
            }
            ensure!(
                results.stats.sync.skipped.is_empty() && results.stats.sync.resources_updated == 0,
                "corpus changed during run"
            );
            let hits = results
                .hits
                .into_iter()
                .map(|hit| {
                    if let Some((_, eligible)) = &pdf_proof {
                        ensure!(
                            eligible.contains_key(&hit.resource),
                            "excluded/unknown PDF hit"
                        );
                    }
                    let hit = ranked(hit, &corpus, &keys)?;
                    if let Some((coverage, _)) = &pdf_proof {
                        pdf_ranked(&hit, &coverage.omitted_text_pages)?;
                    }
                    Ok(hit)
                })
                .collect::<Result<Vec<_>>>()?;
            write_row(
                &mut writer,
                &json!({"type":"query", "qid":query.qid, "mode":name,
                "ranked":collapse(&hits, false), "ranked_pages":if pdf {Some(collapse(&hits,true))} else {None},
                "segments_returned":hits.len(), "query_ms":start.map(|s|s.elapsed().as_secs_f64()*1000.0)}),
            )?;
        }
        if let Some((coverage, eligible)) = &pdf_proof {
            ensure!(
                pdf_policy
                    .as_ref()
                    .context("PDF policy root")?
                    .revalidate()
                    .is_ok_and(|v| v),
                "PDF root identity changed"
            );
            let (mut observed, ids) = pdf_inventory(&root, &corpus, &manifest, &keys, &discovery)?;
            observed.omitted_text_pages = coverage.omitted_text_pages.clone();
            ensure!(
                &observed == coverage && &ids == eligible,
                "PDF corpus changed during mode"
            );
        }
        write_row(
            &mut writer,
            &json!({"type":"complete", "queries":queries.len(), "load_after":load_average(), "wall_seconds":started.map(|s|s.elapsed().as_secs_f64())}),
        )?;
        writer.flush()?;
        fs::rename(partial, &final_path)?;
        eprintln!("wrote {}", final_path.display());
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(all(test, feature = "pdf"))]
#[path = "../crates/seyk-core/tests/support/pdf_fixture.rs"]
mod pdf_fixture;

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory_fixture(
        root: &Path,
        sizes: &[u64],
    ) -> Result<(PathBuf, Vec<u8>, HashSet<String>)> {
        let corpus = root.join("corpus/irpapers-pdf");
        let meta = root.join("meta/irpapers-pdf");
        fs::create_dir_all(&corpus)?;
        fs::create_dir_all(&meta)?;
        let mut files = serde_json::Map::new();
        let mut keys = HashSet::new();
        for (i, size) in sizes.iter().enumerate() {
            let key = blake3::hash((i + 1).to_string().as_bytes()).to_hex()[..16].to_owned();
            let name = format!("{key}.pdf");
            File::create(corpus.join(&name))?.set_len(*size)?;
            files.insert(
                name.clone(),
                json!(
                    blake3::hash(&fs::read(corpus.join(name))?)
                        .to_hex()
                        .to_string()
                ),
            );
            keys.insert(key);
        }
        let mut metadata = serde_json::Map::new();
        for name in ["docmap.tsv", "queries.jsonl", "qrels.tsv"] {
            fs::write(meta.join(name), name)?;
            metadata.insert(
                name.into(),
                json!(blake3::hash(name.as_bytes()).to_hex().to_string()),
            );
        }
        let manifest = serde_json::to_vec(&json!({"files":files,"metadata_blake3":metadata}))?;
        fs::write(meta.join("manifest.json"), &manifest)?;
        Ok((corpus, manifest, keys))
    }

    #[test]
    fn pdf_default_size_boundary_and_inventory_refusals() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().canonicalize()?;
        let options = DiscoveryOptions::default();
        assert_eq!(options.max_file_bytes, PDF_LIMIT);
        let (corpus, manifest, keys) =
            inventory_fixture(&root, &[PDF_LIMIT - 1, PDF_LIMIT, PDF_LIMIT + 1])?;
        let (coverage, eligible) = pdf_inventory(&root, &corpus, &manifest, &keys, &options)?;
        assert_eq!(eligible.len(), 2);
        assert_eq!(coverage.exclusions.len(), 1);
        assert_eq!(coverage.exclusions[0].size_bytes, PDF_LIMIT + 1);
        let mut omitted = options.clone();
        omitted.exclude.push("*.pdf".into());
        assert!(pdf_inventory(&root, &corpus, &manifest, &keys, &omitted).is_err());
        // Equal counts with the wrong identity cannot pass.
        let original = corpus.join(format!("{}.pdf", keys.iter().next().unwrap()));
        let wrong = corpus.join("unknown.pdf");
        fs::rename(&original, &wrong)?;
        assert!(pdf_inventory(&root, &corpus, &manifest, &keys, &options).is_err());
        fs::rename(wrong, &original)?;
        fs::write(root.join("meta/irpapers-pdf/queries.jsonl"), "tampered")?;
        assert!(pdf_inventory(&root, &corpus, &manifest, &keys, &options).is_err());
        fs::write(
            root.join("meta/irpapers-pdf/queries.jsonl"),
            "queries.jsonl",
        )?;
        let mut missing: serde_json::Value = serde_json::from_slice(&manifest)?;
        missing["metadata_blake3"]
            .as_object_mut()
            .unwrap()
            .remove("qrels.tsv");
        let missing = serde_json::to_vec(&missing)?;
        fs::write(root.join("meta/irpapers-pdf/manifest.json"), &missing)?;
        assert!(pdf_inventory(&root, &corpus, &missing, &keys, &options).is_err());
        fs::write(root.join("meta/irpapers-pdf/manifest.json"), &manifest)?;
        File::options().write(true).open(&original)?.set_len(5)?;
        assert!(pdf_inventory(&root, &corpus, &manifest, &keys, &options).is_err());
        // Product root policy runs before inventory reads.
        let denied = root.join(".ssh");
        fs::create_dir(&denied)?;
        assert!(pdf_inventory(&root, &denied, &manifest, &keys, &options).is_err());
        #[cfg(unix)]
        {
            let link = root.join("alias");
            std::os::unix::fs::symlink(&corpus, &link)?;
            assert!(confined(&root, "alias").is_err());
        }
        Ok(())
    }

    #[test]
    fn pdf_diagnostic_classification_and_freshness() -> Result<()> {
        let id = ResourceId::from_path("/public/opaque.pdf")?;
        let eligible = HashMap::from([(id.clone(), "opaque".into())]);
        let mut sync = SyncStats {
            resources_discovered: 1,
            resources_updated: 1,
            page_diagnostics: vec![seyk::PageDiagnostic {
                resource: id,
                page: 2,
                code: SkipCode::Other,
                reason: NO_TEXT.into(),
            }],
            ..Default::default()
        };
        assert_eq!(pdf_diagnostics(&sync, &eligible, true)?[0].page, 2);
        for code in [SkipCode::PdfParseFailed, SkipCode::PdfParserPanicked] {
            sync.page_diagnostics[0].code = code;
            assert_eq!(pdf_diagnostics(&sync, &eligible, true)?.len(), 1);
        }
        sync.page_diagnostics[0].code = SkipCode::Other;
        sync.page_diagnostics[0].reason = "embedded text is not supported by this embedder".into();
        assert!(pdf_diagnostics(&sync, &eligible, true).is_err());
        sync.page_diagnostics[0].reason = NO_TEXT.into();
        sync.page_diagnostics[0].page = 0;
        assert!(pdf_diagnostics(&sync, &eligible, true).is_err());
        sync.page_diagnostics.clear();
        sync.resources_updated = 0;
        sync.resources_unchanged = 1;
        assert!(pdf_diagnostics(&sync, &eligible, true).is_err());
        Ok(())
    }

    #[test]
    fn pdf_parser_misses_preserve_safe_outcomes_and_receipt() -> Result<()> {
        let id = ResourceId::from_path("/public/opaque.pdf")?;
        let eligible = HashMap::from([(id.clone(), "opaque".into())]);
        let mut sync = SyncStats {
            resources_discovered: 1,
            resources_updated: 1,
            page_diagnostics: vec![
                seyk::PageDiagnostic {
                    resource: id.clone(),
                    page: 2,
                    code: SkipCode::PdfParseFailed,
                    reason: "dependency payload".into(),
                },
                seyk::PageDiagnostic {
                    resource: id.clone(),
                    page: 4,
                    code: SkipCode::PdfParserPanicked,
                    reason: "panic payload".into(),
                },
                seyk::PageDiagnostic {
                    resource: id,
                    page: 5,
                    code: SkipCode::Other,
                    reason: NO_TEXT.into(),
                },
            ],
            ..Default::default()
        };
        let outcomes = pdf_diagnostics(&sync, &eligible, true)?;
        assert_eq!(
            outcomes
                .iter()
                .map(|p| (p.page, p.reason.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (2, "pdf_parse_failed"),
                (4, "pdf_parser_panicked"),
                (5, "no_usable_embedded_text")
            ]
        );
        let encoded = serde_json::to_string(&pdf_safe_sync(&sync)?)?;
        assert!(!encoded.contains("payload"));
        for page in [2, 4, 5] {
            assert!(pdf_ranked(&hit("opaque", page), &outcomes).is_err());
        }
        pdf_ranked(&hit("opaque", 3), &outcomes)?;
        assert!(pdf_ranked(&hit("opaque", 0), &outcomes).is_err());
        let owner = PdfOwner::new("synthetic", b"manifest", "profile");
        assert_eq!(owner.proof, "pdf-policy-miss-v1");
        assert_ne!(owner.proof, PDF_PROOF);
        sync.resources_updated = 0;
        sync.resources_unchanged = 1;
        assert_eq!(pdf_diagnostics(&sync, &eligible, false)?, outcomes);
        sync.page_diagnostics[0].code = SkipCode::PdfParserPanicked;
        assert_ne!(pdf_diagnostics(&sync, &eligible, false)?, outcomes);
        sync.resources_updated = 1;
        sync.resources_unchanged = 0;
        sync.page_diagnostics[1].page = 2;
        assert!(pdf_diagnostics(&sync, &eligible, true).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn pdf_task_ownership_refuses_adoption_mismatch_and_second_writer() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().canonicalize()?;
        let profile = SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(
                Dimension::D256,
                "pdf-task-process",
            ))
            .build()?
            .profile_id()
            .to_owned();
        let owner = PdfOwner::new("synthetic", b"manifest", &profile);
        assert!(PdfSession::open(&root, &owner, true).is_err());
        let session = PdfSession::open(&root, &owner, false)?;
        fs::write(&session.cache, b"not opened by ownership tests")?;
        assert!(PdfSession::open(&root, &owner, false).is_err());
        assert!(PdfSession::open(&root, &owner, true).is_err());
        #[cfg(feature = "pdf")]
        {
            fs::create_dir(root.join("corpus"))?;
            fs::write(root.join("manifest"), b"manifest")?;
            let out = child(&root, "locked")?;
            ensure!(
                out.status.success(),
                "second-writer process: {} {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        drop(session);
        for wrong in [
            PdfOwner::new("synthetic", b"changed", &profile),
            PdfOwner::new("synthetic", b"manifest", "other"),
        ] {
            assert!(PdfSession::open(&root, &wrong, true).is_err());
        }
        let mut wrong = PdfOwner::new("synthetic", b"manifest", &profile);
        wrong.proof = PDF_PROOF.into();
        assert!(PdfSession::open(&root, &wrong, true).is_err());
        wrong.proof = PDF_OWNER_PROOF.into();
        wrong.version = 2;
        assert!(PdfSession::open(&root, &wrong, true).is_err());
        let resumed = PdfSession::open(&root, &owner, true)?;
        let dir = resumed.cache.parent().unwrap().to_owned();
        drop(resumed);
        let receipt = fs::read(dir.join("owner.json"))?;
        fs::remove_file(dir.join("owner.json"))?;
        assert!(PdfSession::open(&root, &owner, true).is_err());
        fs::write(dir.join("owner.json"), &receipt)?;
        std::os::unix::fs::symlink(root.join("sentinel"), dir.join("cache.sqlite-wal"))?;
        assert!(PdfSession::open(&root, &owner, true).is_err());
        Ok(())
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_stale_excluded_binding_cannot_supply_hits() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("opaque.pdf");
        fs::write(
            &path,
            pdf_fixture::pdf(&["garden basil"], pdf_fixture::Kind::Plain),
        )?;
        let engine = SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(Dimension::D256, "stale"))
            .build()?;
        engine.index().paths([&path]).execute()?;
        File::options()
            .write(true)
            .open(&path)?
            .set_len(PDF_LIMIT + 1)?;
        for mode in [
            SearchMode::Hybrid,
            SearchMode::Semantic,
            SearchMode::Lexical,
        ] {
            assert!(
                engine
                    .search("basil")
                    .paths([&path])
                    .mode(mode)
                    .execute()?
                    .hits
                    .is_empty()
            );
        }
        Ok(())
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_resource_failures_reject_but_partial_pages_are_misses() -> Result<()> {
        use pdf_fixture::{Kind, pdf};
        let mut cases = vec![b"not a PDF".to_vec()];
        for kind in [
            Kind::Encrypted,
            Kind::MissingContents,
            Kind::BadFlate,
            Kind::Panic,
        ] {
            cases.push(pdf(&["garden first", "basil second"], kind));
        }
        for bytes in cases {
            let tmp = tempfile::tempdir()?;
            let path = tmp.path().canonicalize()?.join("opaque.pdf");
            fs::write(&path, bytes)?;
            let engine = SearchEngine::builder()
                .embedder(common::FakeEmbedder::new(Dimension::D256, "fatal"))
                .build()?;
            let report = engine.index().paths([&path]).execute()?;
            let ids = HashMap::from([(ResourceId::from_path(&path)?, "opaque".into())]);
            assert!(pdf_diagnostics(&report.sync, &ids, true).is_err());
        }
        // BrokenSecond is a genuine page-local panic; corrupt streams above
        // remain whole-resource failures under the unchanged parser contract.
        for (kind, page, reason) in [(Kind::BrokenSecond, 2, "pdf_parser_panicked")] {
            let tmp = tempfile::tempdir()?;
            let path = tmp.path().canonicalize()?.join("opaque.pdf");
            fs::write(
                &path,
                pdf(&["garden first", "healthy second", "basil third"], kind),
            )?;
            let engine = SearchEngine::builder()
                .embedder(common::FakeEmbedder::new(Dimension::D256, "partial"))
                .build()?;
            let indexed = engine.index().paths([&path]).execute()?;
            let ids = HashMap::from([(ResourceId::from_path(&path)?, "opaque".into())]);
            let outcomes = pdf_diagnostics(&indexed.sync, &ids, true)?;
            assert_eq!(
                outcomes,
                vec![OmittedPage {
                    key: "opaque".into(),
                    page,
                    reason: reason.into()
                }]
            );
            for mode in [
                SearchMode::Hybrid,
                SearchMode::Semantic,
                SearchMode::Lexical,
            ] {
                let found = engine
                    .search("garden basil")
                    .paths([&path])
                    .mode(mode)
                    .execute()?;
                assert_eq!(pdf_diagnostics(&found.stats.sync, &ids, false)?, outcomes);
                assert!(!found.hits.is_empty());
                for h in found.hits {
                    let ranked =
                        ranked(h, path.parent().unwrap(), &HashSet::from(["opaque".into()]))?;
                    pdf_ranked(&ranked, &outcomes)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "pdf")]
    type VectorSnapshot = Vec<(String, String, Vec<f32>)>;

    #[cfg(feature = "pdf")]
    fn vectors(engine: &SearchEngine, ids: &HashMap<ResourceId, String>) -> Result<VectorSnapshot> {
        let mut out = engine.with_cache(|c| {
            Ok(
                c.vectors_for(engine.profile_id(), &ids.keys().cloned().collect())?
                    .into_iter()
                    .map(|v| (v.resource.to_string(), v.segment.input_hash, v.vector))
                    .collect::<Vec<_>>(),
            )
        })?;
        out.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        Ok(out)
    }

    #[cfg(feature = "pdf")]
    struct ProcessFake {
        inner: common::FakeEmbedder,
        calls: std::sync::Mutex<Option<PathBuf>>,
    }

    #[cfg(feature = "pdf")]
    impl seyk::Embedder for ProcessFake {
        fn identity(&self) -> &seyk::ModelIdentity {
            seyk::Embedder::identity(&self.inner)
        }
        fn supports(&self, modality: seyk::Modality) -> bool {
            seyk::Embedder::supports(&self.inner, modality)
        }
        fn embed_batch(
            &self,
            requests: &[seyk::EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<seyk::Embedding>> {
            let log = self.calls.lock().unwrap();
            if let Some(path) = log.as_ref() {
                let mut counts: BTreeMap<String, usize> = if path.exists() {
                    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
                } else {
                    BTreeMap::new()
                };
                for request in requests
                    .iter()
                    .filter(|r| r.task == seyk::EmbeddingTask::SearchDocument)
                {
                    let seyk::ResourceInput::Text { title, text } = &request.input else {
                        panic!("text fixture only")
                    };
                    let signature = blake3::hash(&serde_json::to_vec(&(title, text)).unwrap())
                        .to_hex()
                        .to_string();
                    *counts.entry(signature).or_default() += 1;
                }
                fs::write(path, serde_json::to_vec(&counts).unwrap()).unwrap();
            }
            seyk::Embedder::embed_batch(&self.inner, requests)
        }
    }

    #[cfg(feature = "pdf")]
    fn process_engine(
        cache: &Path,
        calls: Option<PathBuf>,
    ) -> Result<(SearchEngine, Arc<ProcessFake>)> {
        let model = Arc::new(ProcessFake {
            inner: common::FakeEmbedder::new(Dimension::D256, "pdf-task-process"),
            calls: std::sync::Mutex::new(calls),
        });
        let engine = SearchEngine::builder()
            .shared_embedder(model.clone())
            .cache(SqliteCache::open(cache)?)
            .build()?;
        Ok((engine, model))
    }

    #[cfg(feature = "pdf")]
    fn child(root: &Path, phase: &str) -> Result<std::process::Output> {
        Ok(Command::new(std::env::current_exe()?)
            .args(["--exact", "tests::pdf_process_child", "--nocapture"])
            .env("SEYK_PDF_TEST_ROOT", root)
            .env("SEYK_PDF_TEST_PHASE", phase)
            .output()?)
    }

    /// Spawned only by the bounded synthetic regression; no Gemma/model loading.
    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_process_child() -> Result<()> {
        let Some(root) = std::env::var_os("SEYK_PDF_TEST_ROOT") else {
            return Ok(());
        };
        let root = PathBuf::from(root);
        let phase = std::env::var("SEYK_PDF_TEST_PHASE")?;
        let corpus = root.join("corpus");
        let ids: HashMap<_, _> = fs::read_dir(&corpus)?
            .map(|e| {
                let path = e?.path();
                Ok((
                    ResourceId::from_path(&path)?,
                    path.file_stem().unwrap().to_str().unwrap().to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        let profile = SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(
                Dimension::D256,
                "pdf-task-process",
            ))
            .build()?
            .profile_id()
            .to_owned();
        let manifest = fs::read(root.join("manifest"))?;
        let owner = PdfOwner::new("synthetic", &manifest, &profile);
        if phase == "locked" {
            ensure!(
                PdfSession::open(&root, &owner, true).is_err(),
                "second writer admitted"
            );
            return Ok(());
        }
        let receipt_path = root.join("cache/irpapers-pdf/tasks/synthetic/owner.json");
        let prior_receipt = fs::read(&receipt_path).ok();
        let session = PdfSession::open(
            &root,
            &owner,
            matches!(phase.as_str(), "negative" | "reset" | "resume"),
        )?;
        let receipt = fs::read(&receipt_path)?;
        assert_eq!(
            serde_json::from_slice::<PdfOwner>(&receipt)?.proof,
            "pdf-policy-miss-v1"
        );
        if let Some(prior) = prior_receipt {
            assert_eq!(receipt, prior);
        }
        let (engine, model) =
            process_engine(&session.cache, Some(root.join("document-input-calls.json")))?;
        let expected: VectorSnapshot = if root.join("expected.json").exists() {
            serde_json::from_slice(&fs::read(root.join("expected.json"))?)?
        } else {
            Vec::new()
        };
        let hashes: Vec<_> = expected.iter().map(|v| v.1.clone()).collect();
        let known = engine.with_cache(|c| c.known_inputs(&profile, &hashes))?;
        if phase != "negative" {
            reset_pdf_bindings(&engine, &ids)?;
            assert!(
                engine
                    .with_cache(|c| c.resource_index(&profile))?
                    .is_empty()
            );
            assert_eq!(
                known,
                engine.with_cache(|c| c.known_inputs(&profile, &hashes))?
            );
        }
        let progress = |event| {
            if phase == "interrupt"
                && matches!(event, seyk::ProgressEvent::Embedding {done, ..} if done > 0)
            {
                let committed = engine
                    .with_cache(|c| c.known_inputs(&profile, &hashes))
                    .unwrap();
                assert!(!committed.is_empty() && committed.len() < hashes.len());
                assert!(
                    engine
                        .with_cache(|c| c.resource_index(&profile))
                        .unwrap()
                        .is_empty()
                );
                fs::write(
                    root.join("checkpoint.json"),
                    serde_json::to_vec(&committed).unwrap(),
                )
                .unwrap();
                // Simulate process termination after SQLite committed vectors, before resource commit.
                std::process::exit(42);
            }
        };
        let indexed = engine
            .index()
            .paths([&corpus])
            .discovery(DiscoveryOptions::default())
            .progress(&progress)
            .execute()?;
        if phase == "negative" {
            assert_eq!(indexed.sync.resources_unchanged, ids.len());
            assert!(indexed.sync.page_diagnostics.is_empty());
            assert!(pdf_diagnostics(&indexed.sync, &ids, true).is_err());
            return Ok(());
        }
        assert_eq!(indexed.resources_removed, 0);
        assert_eq!(indexed.sync.resources_updated, ids.len());
        assert_eq!(indexed.sync.resources_unchanged, 0);
        if phase == "reset" || phase == "resume" {
            assert_eq!(indexed.sync.segments_embedded, hashes.len() - known.len());
            assert_eq!(model.inner.resource_count(), hashes.len() - known.len());
            assert_eq!(vectors(&engine, &ids)?, expected);
        } else {
            fs::write(
                root.join("expected.json"),
                serde_json::to_vec(&vectors(&engine, &ids)?)?,
            )?;
        }
        {
            let diagnostics = pdf_diagnostics(&indexed.sync, &ids, true)?;
            if root.join("parser-failure").exists() {
                assert_eq!(diagnostics.len(), 1);
                assert_eq!(diagnostics[0].reason, "pdf_parser_panicked");
            }
            for mode in [
                SearchMode::Hybrid,
                SearchMode::Semantic,
                SearchMode::Lexical,
            ] {
                let found = engine
                    .search("basil")
                    .paths([&corpus])
                    .mode(mode)
                    .previews(None)
                    .execute()?;
                assert_eq!(
                    pdf_diagnostics(&found.stats.sync, &ids, false)?,
                    diagnostics
                );
                assert!(found.hits.iter().all(|h| ids.contains_key(&h.resource)));
                assert!(
                    found
                        .hits
                        .iter()
                        .any(|h| matches!(h.locator, ResourceLocator::PageText { page: 3, .. }))
                );
            }
            fs::write(
                root.join(format!("{phase}-proof.json")),
                serde_json::to_vec(&json!({"proof":PDF_PROOF,"omitted_text_pages":diagnostics}))?,
            )?;
            if phase != "resume" {
                assert_eq!(
                    diagnostics.iter().map(|p| p.page).collect::<Vec<_>>(),
                    vec![2]
                );
            }
        }
        fs::write(root.join(format!("{phase}-complete")), b"synthetic only")?;
        Ok(())
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn pdf_real_process_reopen_reset_and_interruption() -> Result<()> {
        use pdf_fixture::{Kind, pdf};
        for failure in [false, true] {
            let tmp = tempfile::tempdir()?;
            let root = tmp.path().canonicalize()?;
            fs::create_dir(root.join("corpus"))?;
            let bytes = pdf(
                &["garden first", "", "basil third"],
                if failure {
                    Kind::BrokenSecond
                } else {
                    Kind::Plain
                },
            );
            fs::write(root.join("corpus/opaque.pdf"), &bytes)?;
            fs::write(root.join("manifest"), blake3::hash(&bytes).as_bytes())?;
            if failure {
                fs::write(root.join("parser-failure"), b"expected")?;
            }
            let sentinel = root.join("shared.sqlite");
            drop(SqliteCache::open(&sentinel)?);
            let sentinel_bytes = fs::read(&sentinel)?;
            for phase in ["seed", "negative", "reset"] {
                let output = child(&root, phase)?;
                ensure!(
                    output.status.success(),
                    "{phase}: {} {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            assert_eq!(fs::read(&sentinel)?, sentinel_bytes);
            assert_eq!(fs::read(root.join("corpus/opaque.pdf"))?, bytes);
        }
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().canonicalize()?;
        let corpus = root.join("corpus");
        fs::create_dir(&corpus)?;
        let bytes = pdf(
            &[
                "garden one",
                "harbor two",
                "basil three",
                "orchard four",
                "river five",
                "meadow six",
            ],
            Kind::Plain,
        );
        let path = corpus.join("opaque.pdf");
        fs::write(&path, &bytes)?;
        fs::write(root.join("manifest"), blake3::hash(&bytes).as_bytes())?;
        let ids = HashMap::from([(ResourceId::from_path(&path)?, "opaque".into())]);
        let (reference, _) = process_engine(&root.join("reference.sqlite"), None)?;
        reference.index().paths([&corpus]).execute()?;
        let expected = vectors(&reference, &ids)?;
        fs::write(root.join("expected.json"), serde_json::to_vec(&expected)?)?;
        let interrupted = child(&root, "interrupt")?;
        assert_eq!(
            interrupted.status.code(),
            Some(42),
            "{}",
            String::from_utf8_lossy(&interrupted.stdout)
        );
        assert!(!root.join("interrupt-complete").exists());
        let committed: HashSet<String> =
            serde_json::from_slice(&fs::read(root.join("checkpoint.json"))?)?;
        assert!(committed.len() > 0 && committed.len() < expected.len());
        let resumed = child(&root, "resume")?;
        ensure!(
            resumed.status.success(),
            "resume: {} {}",
            String::from_utf8_lossy(&resumed.stdout),
            String::from_utf8_lossy(&resumed.stderr)
        );
        let calls: BTreeMap<String, usize> =
            serde_json::from_slice(&fs::read(root.join("document-input-calls.json"))?)?;
        assert_eq!(calls.len(), expected.len());
        assert!(
            calls.values().all(|count| *count == 1),
            "a committed input was re-inferred"
        );
        Ok(())
    }
    fn hit(key: &str, page: u32) -> Ranked {
        Ranked {
            key: key.into(),
            rank: 0,
            score: 0.5,
            semantic: Some(0.8),
            lexical: None,
            filename: false,
            locator: ResourceLocator::PageText {
                page,
                byte_start: 0,
                byte_end: 1,
                text_version: "v1".into(),
                page_text_hash: "hash".into(),
            },
        }
    }
    #[test]
    fn warm_sqlite_rankings_are_deterministic() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let corpus = tmp.path().join("corpus");
        fs::create_dir(&corpus)?;
        let corpus = corpus.canonicalize()?;
        for (name, text) in [
            ("a", "alpha beta gamma"),
            ("b", "alpha delta"),
            ("c", "beta gamma"),
        ] {
            fs::write(corpus.join(format!("{name}.txt")), text)?;
        }
        let cache = tmp.path().join("dedicated.sqlite");
        let mut runs = Vec::new();
        for pass in 0..2 {
            let engine = SearchEngine::builder()
                .embedder(common::FakeEmbedder::new(
                    Dimension::D256,
                    "bench-determinism",
                ))
                .cache(SqliteCache::open(&cache)?)
                .build()?;
            let report = engine.index().paths([&corpus]).execute()?;
            if pass == 1 {
                assert_eq!(report.sync.segments_embedded, 0);
            }
            let keys = ["a", "b", "c"].map(String::from).into_iter().collect();
            let mut run = Vec::new();
            for mode in [
                SearchMode::Hybrid,
                SearchMode::Semantic,
                SearchMode::Lexical,
            ] {
                for query in ["alpha", "beta gamma", "delta"] {
                    let result = engine
                        .search(query)
                        .mode(mode)
                        .paths([&corpus])
                        .previews(None)
                        .execute()?;
                    let hits = result
                        .hits
                        .into_iter()
                        .map(|h| ranked(h, &corpus, &keys))
                        .collect::<Result<Vec<_>>>()?;
                    run.push(serde_json::to_vec(&collapse(&hits, false))?);
                }
            }
            runs.push(run);
        }
        assert_eq!(runs[0], runs[1]);
        Ok(())
    }

    #[test]
    fn collapse_preserves_first_seen_and_best_locator() {
        let hits = vec![
            hit("b", 2),
            hit("a", 1),
            hit("b", 3),
            hit("a", 1),
            hit("c", 1),
        ];
        let result = collapse(&hits, false);
        assert_eq!(
            result
                .iter()
                .map(|h| (h.key.as_str(), h.rank))
                .collect::<Vec<_>>(),
            vec![("b", 1), ("a", 2), ("c", 3)]
        );
        assert_eq!(result[0].locator, hits[0].locator);
        assert_eq!(
            collapse(&hits, true)
                .iter()
                .map(|h| h.rank)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(collapse(&[], false).is_empty());
    }
}
