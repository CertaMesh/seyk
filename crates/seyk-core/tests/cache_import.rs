mod common;

use common::FakeEmbedder;
use rusqlite::{Connection, params};
use seyk_core::cache::CacheProfile;
use seyk_core::{
    CacheImportMapping, Dimension, EmbeddingCache, ResourceId, SearchEngine, SqliteCache,
};
use std::{collections::HashSet, fs, path::PathBuf, sync::Arc};

const BODY: &str = "Synthetic violet compass gardens alpha violet. A quiet amber lighthouse.";

struct Fixture {
    _temp: tempfile::TempDir,
    source: PathBuf,
    destination: PathBuf,
    old: PathBuf,
    new: PathBuf,
    profile: CacheProfile,
    fake: Arc<FakeEmbedder>,
    // Keep the source open to exercise a committed WAL snapshot.
    engine: SearchEngine,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let old = root.join("oldepochtoken/note.md");
        let new = root.join("newepochtoken/note.md");
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        fs::create_dir_all(new.parent().unwrap()).unwrap();
        fs::write(&old, BODY).unwrap();
        fs::write(&new, BODY).unwrap();
        let source = root.join("source.sqlite");
        let destination = root.join("destination.sqlite");
        let fake = Arc::new(FakeEmbedder::new(Dimension::D128, "import"));
        let engine = SearchEngine::builder()
            .shared_embedder(fake.clone())
            .cache(SqliteCache::open(&source).unwrap())
            .build()
            .unwrap();
        engine.index().paths([&old]).execute().unwrap();
        let profile = engine.profile().clone();
        Self {
            _temp: temp,
            source,
            destination,
            old,
            new,
            profile,
            fake,
            engine,
        }
    }
    fn mapping(&self) -> CacheImportMapping {
        CacheImportMapping {
            version: 1,
            old: ResourceId::from_path(&self.old).unwrap(),
            new: ResourceId::from_path(&self.new).unwrap(),
            content_hash: blake3::hash(BODY.as_bytes()).to_hex().to_string(),
        }
    }
    fn import(&self) -> seyk_core::Result<seyk_core::CacheImportReport> {
        SqliteCache::import_from(
            &self.source,
            &self.destination,
            &self.profile,
            &[self.mapping()],
        )
    }
    fn sql(&self, sql: &str) {
        Connection::open(&self.source)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
    }
    fn fts(&self) -> String {
        let conn = Connection::open(&self.source).unwrap();
        conn.query_row(
            "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%'",
            [],
            |r| r.get(0),
        )
        .unwrap()
    }
    fn assert_empty_destination(&self) {
        let cache = SqliteCache::open(&self.destination).unwrap();
        assert!(cache.stats().unwrap().profiles.is_empty());
    }
}

#[test]
fn parent_epoch_move_reuses_vectors_and_evidence_without_embedding() {
    let f = Fixture::new();
    let old_id = ResourceId::from_path(&f.old).unwrap();
    let new_id = ResourceId::from_path(&f.new).unwrap();
    let old_cache = SqliteCache::open(&f.source).unwrap();
    let old_vectors = old_cache
        .vectors_for(&f.profile.id(), &HashSet::from([old_id]))
        .unwrap();
    fs::remove_file(&f.old).unwrap(); // The old path must never be read.
    let report = f.import().unwrap();
    assert_eq!(report.resources, 1);
    assert_eq!(
        report.drops.keys().map(String::as_str).collect::<Vec<_>>(),
        vec![
            "hash_mismatch",
            "missing_hash",
            "preparation_changed",
            "unavailable_or_unstable",
            "unmapped"
        ]
    );
    assert!(report.drops.values().all(|count| *count == 0));
    assert_eq!(report.segments, report.fts_rows);
    assert!(report.vectors > 0);
    let imported = SqliteCache::open(&f.destination).unwrap();
    let new_vectors = imported
        .vectors_for(&f.profile.id(), &HashSet::from([new_id.clone()]))
        .unwrap();
    assert_eq!(old_vectors.len(), new_vectors.len());
    for (a, b) in old_vectors.iter().zip(&new_vectors) {
        assert_eq!(a.segment, b.segment);
        assert_eq!(
            a.vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(imported.resource_index(&f.profile.id()).unwrap().len(), 1);
    let engine = SearchEngine::builder()
        .shared_embedder(f.fake.clone())
        .cache(imported)
        .build()
        .unwrap();
    f.fake.reset_resource_count();
    engine.index().paths([&f.new]).execute().unwrap();
    assert_eq!(f.fake.resource_count(), 0);
    let hits = engine
        .search("violet")
        .paths([&f.new])
        .execute()
        .unwrap()
        .hits;
    assert!(!hits.is_empty());
    for hit in &hits {
        assert_eq!(hit.resource, new_id);
        engine.extract(hit).unwrap();
    }
    // Stored content and on-disk bytes never retain source word order or old path labels.
    let conn = Connection::open(&f.destination).unwrap();
    let (path, body): (String, String) = conn
        .query_row(&format!("SELECT path,body FROM \"{}\"", f.fts()), [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert!(path.contains("newepochtoken"));
    assert!(!path.contains("oldepochtoken"));
    for (term, expected) in [("newepochtoken", 1), ("oldepochtoken", 0)] {
        let count: i64 = conn
            .query_row(
                &format!("SELECT count(*) FROM \"{}\" WHERE path MATCH ?1", f.fts()),
                [term],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, expected);
    }
    assert_eq!(
        body.split_whitespace().filter(|t| *t == "violet").count(),
        2
    );
    assert!(!body.contains(BODY));
    for suffix in ["", "-wal"] {
        if let Ok(bytes) = fs::read(format!("{}{suffix}", f.destination.display())) {
            assert!(!bytes.windows(BODY.len()).any(|w| w == BODY.as_bytes()));
            assert!(
                !bytes
                    .windows(b"oldepochtoken".len())
                    .any(|w| w == b"oldepochtoken")
            );
        }
    }
    fs::write(&f.new, "changed").unwrap();
    assert!(engine.extract(&hits[0]).is_err());
    assert_eq!(f.engine.profile(), &f.profile);
}

#[test]
fn changed_hashes_missing_paths_and_preparation_changes_drop_whole_resources() {
    for change in [
        "expected",
        "observed",
        "cached",
        "missing",
        "stem",
        "extension",
        "descriptor",
        "unmapped",
    ] {
        let f = Fixture::new();
        let mut m = f.mapping();
        match change {
            "expected" => m.content_hash = "0".repeat(64),
            "observed" => fs::write(&f.new, "different bytes").unwrap(),
            "cached" => f.sql(&format!(
                "UPDATE resources SET content_hash='{}'",
                "0".repeat(64)
            )),
            "missing" => fs::remove_file(&f.new).unwrap(),
            "stem" | "extension" => {
                let new = f.new.with_file_name(if change == "stem" {
                    "renamed.md"
                } else {
                    "note.txt"
                });
                fs::copy(&f.new, &new).unwrap();
                m.new = ResourceId::from_path(new).unwrap();
            }
            "descriptor" => f.sql("UPDATE resources SET descriptor_hash=NULL"),
            _ => {}
        }
        let mappings = if change == "unmapped" {
            vec![]
        } else {
            vec![m]
        };
        let r = SqliteCache::import_from(&f.source, &f.destination, &f.profile, &mappings).unwrap();
        assert_eq!(
            (r.resources, r.segments, r.vectors, r.fts_rows),
            (0, 0, 0, 0),
            "{change}"
        );
        assert_eq!(r.drops.values().sum::<u64>(), 1, "{change}");
        let reason = match change {
            "expected" | "observed" | "cached" => "hash_mismatch",
            "missing" => "unavailable_or_unstable",
            "unmapped" => "unmapped",
            _ => "preparation_changed",
        };
        assert_eq!(r.drops[reason], 1, "{change}");
        let cache = SqliteCache::open(&f.destination).unwrap();
        assert!(cache.resource_index(&f.profile.id()).unwrap().is_empty());
    }
}

#[test]
fn corrupt_evidence_rolls_back_and_retry_succeeds() {
    for corruption in [
        "vector",
        "missing_vector",
        "locator",
        "locator_extra",
        "page_text_version",
        "input_hash",
        "modality",
        "media_type",
        "fts_missing",
        "fts_order",
        "ordinal",
        "profile",
        "dimension",
    ] {
        let f = Fixture::new();
        let sql = match corruption {
            "vector" => "UPDATE embeddings SET vector=x'00'".into(),
            "missing_vector" => "DELETE FROM embeddings".into(),
            "locator" => "UPDATE segments SET locator='{}'".into(),
            "locator_extra" => "UPDATE segments SET locator=json_set(locator,'$.unexpected','synthetic old epoch source-order sentinel')".into(),
            "page_text_version" => format!("UPDATE segments SET locator=json_object('kind','page_text','page',1,'byte_start',0,'byte_end',1,'text_version','synthetic source order sentinel','page_text_hash','{}')", "0".repeat(64)),
            "input_hash" => "UPDATE segments SET input_hash='synthetic source order sentinel'; UPDATE embeddings SET input_hash='synthetic source order sentinel'".into(),
            "modality" => "UPDATE resources SET modality='synthetic source order sentinel'".into(),
            "media_type" => "UPDATE resources SET media_type='synthetic source order sentinel'".into(),
            "fts_missing" => format!("DELETE FROM \"{}\"", f.fts()),
            "fts_order" => format!("UPDATE \"{}\" SET body='zulu alpha'", f.fts()),
            "ordinal" => "UPDATE segments SET ordinal=10".into(),
            "profile" => "UPDATE profiles SET profile='{}'".into(),
            _ => "UPDATE profiles SET dimension=256".into(),
        };
        f.sql(&sql);
        assert!(f.import().is_err(), "{corruption}");
        f.assert_empty_destination();
    }
}

#[test]
fn mappings_reject_duplicates_unknown_fields_versions_and_nonlocal_ids() {
    let f = Fixture::new();
    let m = serde_json::to_value(f.mapping()).unwrap();
    for (key, value) in [
        ("version", serde_json::json!(2)),
        ("extra", serde_json::json!(true)),
        ("content_hash", serde_json::json!("ABC")),
        ("new", serde_json::json!("mem://other/note")),
    ] {
        let mut bad = m.clone();
        bad[key] = value;
        assert!(
            CacheImportMapping::read_jsonl(serde_json::to_string(&bad).unwrap().as_bytes())
                .is_err()
        );
    }
    let line = serde_json::to_string(&m).unwrap();
    assert!(CacheImportMapping::read_jsonl(format!("{line}\n{line}\n").as_bytes()).is_err());
    assert!(
        SqliteCache::import_from(
            &f.source,
            &f.destination,
            &f.profile,
            &[f.mapping(), f.mapping()]
        )
        .is_err()
    );
    let mut other = f.mapping();
    other.old = ResourceId::from_path(f.old.with_file_name("other.md")).unwrap();
    assert!(
        SqliteCache::import_from(&f.source, &f.destination, &f.profile, &[f.mapping(), other])
            .is_err()
    );
    assert_eq!(
        CacheImportMapping::read_jsonl(format!("\n{line}\n\n").as_bytes())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn source_schema_aliases_and_populated_destination_are_rejected_without_mutation() {
    for version in ["1", "2", "3", "4", "5", "7", ""] {
        let f = Fixture::new();
        f.sql(&format!(
            "UPDATE meta SET value='{version}' WHERE key='schema_version'"
        ));
        assert!(f.import().is_err());
        assert!(!f.destination.exists());
        let stored: String = Connection::open(&f.source)
            .unwrap()
            .query_row(
                "SELECT value FROM meta WHERE key='schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, version);
    }
    let f = Fixture::new();
    assert!(SqliteCache::import_from(&f.source, &f.source, &f.profile, &[f.mapping()]).is_err());
    #[cfg(unix)]
    {
        fs::hard_link(&f.source, &f.destination).unwrap();
        assert!(f.import().is_err());
        fs::remove_file(&f.destination).unwrap();
    }
    f.import().unwrap();
    let before = SqliteCache::open(&f.destination)
        .unwrap()
        .stats()
        .unwrap()
        .profiles;
    assert!(f.import().is_err());
    assert_eq!(
        SqliteCache::open(&f.destination)
            .unwrap()
            .stats()
            .unwrap()
            .profiles,
        before
    );
}

#[test]
fn only_referenced_vectors_from_selected_profile_are_copied() {
    let f = Fixture::new();
    let mut cache = SqliteCache::open(&f.source).unwrap();
    cache
        .put_embeddings(&f.profile.id(), &[("orphan".into(), vec![0.0; 128])])
        .unwrap();
    let mut other = f.profile.clone();
    other.segmenter_version.push_str("other");
    cache.register_profile(&other.id(), &other).unwrap();
    cache
        .put_embeddings(&other.id(), &[("other".into(), vec![0.0; 128])])
        .unwrap();
    let mut unmatched = f.mapping();
    unmatched.old = ResourceId::from_path(f.old.with_file_name("absent.md")).unwrap();
    unmatched.new = ResourceId::from_path(f.new.with_file_name("absent.md")).unwrap();
    let report = SqliteCache::import_from(
        &f.source,
        &f.destination,
        &f.profile,
        &[f.mapping(), unmatched],
    )
    .unwrap();
    assert_eq!(report.unmatched_mappings, 1);
    let dst = Connection::open(&f.destination).unwrap();
    let n: i64 = dst
        .query_row(
            "SELECT count(*) FROM embeddings WHERE input_hash IN ('orphan','other')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0);
    assert_eq!(
        SqliteCache::open(&f.destination)
            .unwrap()
            .stats()
            .unwrap()
            .profiles
            .len(),
        1
    );
    let source_resources: i64 = Connection::open(&f.source)
        .unwrap()
        .query_row(
            "SELECT count(*) FROM resources WHERE profile=?1",
            params![f.profile.id()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(source_resources, 1);
}

#[cfg(unix)]
#[test]
fn symlinks_and_disallowed_names_drop_without_import() {
    for denied in [false, true] {
        let f = Fixture::new();
        let mut m = f.mapping();
        if denied {
            let secret = f.new.parent().unwrap().join(".env");
            fs::copy(&f.new, &secret).unwrap();
            m.new = ResourceId::from_path(secret).unwrap();
        } else {
            fs::remove_file(&f.new).unwrap();
            std::os::unix::fs::symlink(&f.old, &f.new).unwrap();
        }
        let report = SqliteCache::import_from(&f.source, &f.destination, &f.profile, &[m]).unwrap();
        assert_eq!(report.resources, 0);
        assert_eq!(report.drops.values().sum::<u64>(), 1);
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_parent_paths_round_trip() {
    use std::os::unix::ffi::OsStringExt;
    let f = Fixture::new();
    let parent = f
        .new
        .parent()
        .unwrap()
        .join(std::ffi::OsString::from_vec(b"epoch-\xff".to_vec()));
    let mut m = f.mapping();
    // The old path need not exist, so this identity works even on UTF-8-only filesystems.
    let old = parent.join("note.md");
    let old_id = ResourceId::from_path(&old).unwrap();
    use std::os::unix::ffi::OsStrExt;
    let conn = Connection::open(&f.source).unwrap();
    conn.execute_batch("BEGIN; PRAGMA defer_foreign_keys=ON;")
        .unwrap();
    conn.execute("UPDATE resources SET path=?1", [old.as_os_str().as_bytes()])
        .unwrap();
    conn.execute("UPDATE segments SET path=?1", [old.as_os_str().as_bytes()])
        .unwrap();
    conn.execute_batch("COMMIT").unwrap();
    m.old = old_id;
    #[cfg(target_os = "linux")]
    {
        fs::create_dir(&parent).unwrap();
        fs::copy(&f.new, &old).unwrap();
        m.new = ResourceId::from_path(&old).unwrap();
    }
    let report =
        SqliteCache::import_from(&f.source, &f.destination, &f.profile, &[m.clone()]).unwrap();
    assert_eq!(report.resources, 1);
    assert!(
        SqliteCache::open(&f.destination)
            .unwrap()
            .resource(&f.profile.id(), &m.new)
            .unwrap()
            .is_some()
    );
}

#[test]
fn shared_inputs_are_copied_once_and_mixed_drops_commit() {
    let f = Fixture::new();
    let old2 = f.old.parent().unwrap().join("second/note.md");
    let new2 = f.new.parent().unwrap().join("second/note.md");
    fs::create_dir_all(old2.parent().unwrap()).unwrap();
    fs::create_dir_all(new2.parent().unwrap()).unwrap();
    fs::write(&old2, BODY).unwrap();
    fs::write(&new2, BODY).unwrap();
    f.engine.index().paths([&old2]).execute().unwrap();
    let mut m2 = f.mapping();
    m2.old = ResourceId::from_path(&old2).unwrap();
    m2.new = ResourceId::from_path(&new2).unwrap();
    let old3 = f.old.with_file_name("changed.md");
    let new3 = f.new.with_file_name("changed.md");
    fs::write(&old3, BODY).unwrap();
    fs::write(&new3, "different content").unwrap();
    f.engine.index().paths([&old3]).execute().unwrap();
    let mut m3 = f.mapping();
    m3.old = ResourceId::from_path(&old3).unwrap();
    m3.new = ResourceId::from_path(&new3).unwrap();
    let report = SqliteCache::import_from(
        &f.source,
        &f.destination,
        &f.profile,
        &[f.mapping(), m2, m3],
    )
    .unwrap();
    assert_eq!(report.resources, 2);
    assert_eq!(report.drops["hash_mismatch"], 1);
    assert_eq!(report.vectors * 2, report.segments);
}

#[test]
fn malformed_unversioned_source_is_never_initialized() {
    let f = Fixture::new();
    let source = f.source.with_file_name("foreign.sqlite");
    Connection::open(&source)
        .unwrap()
        .execute_batch(
            "CREATE TABLE foreign_data(value TEXT); INSERT INTO foreign_data VALUES('synthetic');",
        )
        .unwrap();
    let before = fs::read(&source).unwrap();
    assert!(SqliteCache::import_from(&source, &f.destination, &f.profile, &[f.mapping()]).is_err());
    assert_eq!(fs::read(source).unwrap(), before);
    assert!(!f.destination.exists());
}

#[test]
fn empty_destination_rejects_views_and_triggers_without_importing() {
    for object in [
        "CREATE VIEW unexpected_view AS SELECT 1",
        "CREATE TRIGGER unexpected_trigger AFTER INSERT ON profiles BEGIN UPDATE profiles SET dimension=1; END",
    ] {
        let f = Fixture::new();
        drop(SqliteCache::open(&f.destination).unwrap());
        let conn = Connection::open(&f.destination).unwrap();
        conn.execute_batch(object).unwrap();
        assert!(
            f.import()
                .unwrap_err()
                .to_string()
                .contains("foreign schema objects")
        );
        f.assert_empty_destination();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM resources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn unknown_profile_has_a_source_specific_error() {
    let f = Fixture::new();
    assert!(
        SqliteCache::import_profile(&f.source, "missing")
            .unwrap_err()
            .to_string()
            .contains("profile not found in source")
    );
    assert!(!f.destination.exists());
}
