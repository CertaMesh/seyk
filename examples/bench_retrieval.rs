//! Public retrieval harness. See docs/benchmarks.md. Never uses the default cache.
use std::collections::HashSet;
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
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
use seyk::{Dimension, ResourceLocator, SearchEngine, SearchHit, SearchMode, SqliteCache};

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
    /// Diagnostic only; never changes the engine default.
    #[arg(long)]
    lexical_weight: Option<f32>,
    /// Separate warm reruns without overwriting their reference run.
    #[arg(long, default_value = "run")]
    label: String,
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
    let root = dirs::home_dir()
        .context("no home directory")?
        .join(".cache/seyk-bench");
    ensure!(
        !root.symlink_metadata()?.file_type().is_symlink(),
        "benchmark root must not be a symlink"
    );
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
    let keys: HashSet<String> = docmap
        .lines()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap_or("").to_owned())
        .collect();
    ensure!(!keys.is_empty(), "empty docmap");
    let pdf = args.ds == "irpapers-pdf";
    ensure!(
        !pdf || cfg!(feature = "pdf"),
        "PDF track requires the pdf feature"
    );
    let extension = if pdf { "pdf" } else { "txt" };
    let mut count = 0;
    for entry in fs::read_dir(&corpus)? {
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
    ensure!(count == keys.len(), "incomplete corpus");
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
    let started = Instant::now();
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
    let cache = confined(&root, &format!("cache/{}/{profile_id}.sqlite", args.ds))?;
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
    let before_stats = engine.with_cache(|c| c.stats())?;
    eprintln!(
        "indexing {count} resources; embedding progress reports segments needed before inference"
    );
    let progress = |event| eprintln!("{event:?}");
    let index = engine
        .index()
        .paths([&corpus])
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
        *diagnostics
            .entry((&page.resource, page.code))
            .or_insert(0usize) += 1;
    }
    for ((resource, code), pages) in diagnostics {
        let path = resource.to_file_path().context("non-file diagnostic")?;
        let hash = blake3::hash(&fs::read(path)?);
        eprintln!("page diagnostic: hash={hash} pages={pages} code={code:?}");
    }
    ensure!(
        index.sync.resources_discovered == count,
        "discovery omitted benchmark resources"
    );
    if !index.sync.skipped.is_empty() {
        bail!(
            "{} resources skipped; refusing incomplete run",
            index.sync.skipped.len()
        );
    }
    let after_stats = engine.with_cache(|c| c.stats())?;
    let env = json!({"type":"env", "schema":1, "dataset":args.ds, "dataset_manifest":provenance,
        "manifest_blake3":blake3::hash(&manifest).to_hex().to_string(),
        "git_sha":git_sha, "dirty":dirty, "run_id":run_id, "profile_id":profile_id,
        "profile":engine.profile(), "precision":args.precision, "dimension":args.dimension,
        "threads":args.threads, "depth":args.depth, "lexical_weight":args.lexical_weight,
        "os":std::env::consts::OS, "arch":std::env::consts::ARCH,
        "logical_cpus":std::thread::available_parallelism().ok().map(|v|v.get()),
        "rust":command("rustc", &["--version"] )?, "load_before":load_before,
        "load_after_index":load_average(), "index":index,
        "cache_before":before_stats, "cache_after":after_stats});
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
            let start = Instant::now();
            let mut request = engine
                .search(&query.text)
                .paths([&corpus])
                .mode(mode)
                .limit(args.depth)
                .previews(None);
            if let Some(w) = args.lexical_weight {
                request = request.lexical_weight(w);
            }
            let results = request.execute()?;
            ensure!(
                results.stats.sync.skipped.is_empty() && results.stats.sync.resources_updated == 0,
                "corpus changed during run"
            );
            let hits = results
                .hits
                .into_iter()
                .map(|hit| ranked(hit, &corpus, &keys))
                .collect::<Result<Vec<_>>>()?;
            write_row(
                &mut writer,
                &json!({"type":"query", "qid":query.qid, "mode":name,
                "ranked":collapse(&hits, false), "ranked_pages":if pdf {Some(collapse(&hits,true))} else {None},
                "segments_returned":hits.len(), "query_ms":start.elapsed().as_secs_f64()*1000.0}),
            )?;
        }
        write_row(
            &mut writer,
            &json!({"type":"complete", "queries":queries.len(), "load_after":load_average(), "wall_seconds":started.elapsed().as_secs_f64()}),
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

#[cfg(test)]
mod tests {
    use super::*;
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
