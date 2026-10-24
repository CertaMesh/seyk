#![cfg(feature = "cli")]
mod common;
use assert_cmd::Command;
use common::FakeEmbedder;
use seyk::{CacheImportMapping, Dimension, ResourceId, SearchEngine, SqliteCache};
use std::{fs, path::PathBuf};

struct Fixture {
    _temp: tempfile::TempDir,
    source: PathBuf,
    destination: PathBuf,
    map: PathBuf,
    profile: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let old = root.join("old/note.md");
        let new = root.join("new/note.md");
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        fs::create_dir_all(new.parent().unwrap()).unwrap();
        let body = "Synthetic violet amber compass garden.";
        fs::write(&old, body).unwrap();
        fs::write(&new, body).unwrap();
        let source = root.join("source.sqlite");
        let destination = root.join("new.sqlite");
        let map = root.join("mapping.jsonl");
        let engine = SearchEngine::builder()
            .embedder(FakeEmbedder::new(Dimension::D128, "cli-import"))
            .cache(SqliteCache::open(&source).unwrap())
            .build()
            .unwrap();
        engine.index().paths([&old]).execute().unwrap();
        let profile = engine.profile_id().to_string();
        let mapping = CacheImportMapping {
            version: 1,
            old: ResourceId::from_path(old).unwrap(),
            new: ResourceId::from_path(new).unwrap(),
            content_hash: blake3::hash(body.as_bytes()).to_hex().to_string(),
        };
        fs::write(
            &map,
            format!("{}\n", serde_json::to_string(&mapping).unwrap()),
        )
        .unwrap();
        Self {
            _temp: temp,
            source,
            destination,
            map,
            profile,
        }
    }
    fn command(&self) -> Command {
        let mut cmd = Command::cargo_bin("seyk").unwrap();
        cmd.args(["cache", "import", "--from"])
            .arg(&self.source)
            .arg("--map")
            .arg(&self.map)
            .arg("--profile")
            .arg(&self.profile)
            .args(["--json"]);
        cmd.env_remove("SEYK_CACHE");
        cmd
    }
}

#[test]
fn offline_import_reports_success_and_rejects_overwrite() {
    let f = Fixture::new();
    let output = f
        .command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["resources"], 1);
    assert_eq!(report["segments"], report["fts_rows"]);
    f.command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .failure();
}

#[test]
fn import_requires_explicit_destination_even_with_environment_default() {
    let f = Fixture::new();
    f.command().assert().failure();
    assert!(!f.destination.exists());
    f.command()
        .env("SEYK_CACHE", &f.destination)
        .assert()
        .failure();
    assert!(!f.destination.exists());
}

#[test]
fn invalid_jsonl_is_private_and_retry_after_failure_works() {
    let f = Fixture::new();
    let good = fs::read(&f.map).unwrap();
    fs::write(&f.map, "secret-source-sentinel-not-json").unwrap();
    let output = f
        .command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .failure()
        .get_output()
        .clone();
    assert!(!String::from_utf8_lossy(&output.stderr).contains("secret-source-sentinel"));
    assert!(!f.destination.exists());
    fs::write(&f.map, good).unwrap();
    f.command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .success();
}

#[test]
fn mismatch_is_successful_filtering_and_profile_is_required() {
    let f = Fixture::new();
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&f.map).unwrap()).unwrap();
    value["content_hash"] = serde_json::json!("0".repeat(64));
    fs::write(&f.map, serde_json::to_string(&value).unwrap()).unwrap();
    let output = f
        .command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(report["resources"], 0);
    assert_eq!(report["drops"]["hash_mismatch"], 1);
    Command::cargo_bin("seyk")
        .unwrap()
        .args(["cache", "import", "--from"])
        .arg(&f.source)
        .arg("--map")
        .arg(&f.map)
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .failure();
}

#[test]
fn help_explains_offline_source_and_mapping_requirements() {
    Command::cargo_bin("seyk")
        .unwrap()
        .args(["cache", "import", "--help"])
        .assert()
        .success();
}

#[test]
fn corrupt_locator_rolls_back_cli_import_and_repaired_source_can_retry() {
    use seyk::{EmbeddingCache, ResourceLocator, ResourceWrite, SegmentWrite};
    let f = Fixture::new();
    let mapping: CacheImportMapping = serde_json::from_slice(&fs::read(&f.map).unwrap()).unwrap();
    let mut source = SqliteCache::open(&f.source).unwrap();
    let mut resource = source.resource(&f.profile, &mapping.old).unwrap().unwrap();
    let original = resource.segments[0].locator.clone();
    resource.segments[0].locator = ResourceLocator::Text {
        byte_start: 0,
        byte_end: resource.metadata.size_bytes + 1,
        line_start: None,
        line_end: None,
    };
    let write = |resource: &seyk::cache::CachedResource, source: &mut SqliteCache| {
        source
            .put_resource(
                &f.profile,
                &mapping.old,
                &ResourceWrite {
                    metadata: &resource.metadata,
                    version: &resource.version,
                    descriptor_hash: resource.descriptor_hash.as_deref(),
                    segments: resource
                        .segments
                        .iter()
                        .map(|segment| SegmentWrite {
                            segment,
                            lexical_body: "Synthetic violet amber compass garden.",
                        })
                        .collect(),
                },
            )
            .unwrap();
    };
    write(&resource, &mut source);
    f.command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .failure();
    assert!(
        SqliteCache::open(&f.destination)
            .unwrap()
            .stats()
            .unwrap()
            .profiles
            .is_empty()
    );
    resource.segments[0].locator = original;
    write(&resource, &mut source);
    f.command()
        .arg("--cache-path")
        .arg(&f.destination)
        .assert()
        .success();
}
