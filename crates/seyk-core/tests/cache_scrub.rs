//! Bundled-SQLite scrub measurement. All content and files are synthetic/temporary.
mod common;

use common::FakeEmbedder;
use rusqlite::{Connection, params};
use seyk_core::{Dimension, Embedder, EmbeddingCache, SqliteCache, cache::CacheProfile};
use std::{fs, time::Instant};

#[test]
#[ignore = "100k synthetic FTS segments; release-mode privacy scrub cost measurement"]
fn measure_scrub_at_100k_segments() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("cache.sqlite3");
    let mut cache = SqliteCache::open(&path).unwrap();
    cache
        .register_profile(
            "p",
            &CacheProfile {
                model: FakeEmbedder::new(Dimension::D128, "scrub-cost")
                    .identity()
                    .clone(),
                segmenter_version: "synthetic".into(),
            },
        )
        .unwrap();
    let mut db = Connection::open(&path).unwrap();
    db.execute_batch(
        "PRAGMA secure_delete=ON; PRAGMA synchronous=NORMAL; PRAGMA wal_autocheckpoint=0;",
    )
    .unwrap();
    let words = (0..200)
        .map(|i| format!("syntheticword{i:03}"))
        .collect::<Vec<_>>()
        .join(" ");
    let tx = db.transaction().unwrap();
    {
        let version =
            serde_json::to_string(&seyk_core::ResourceVersion::known("synthetic:1")).unwrap();
        let locator = serde_json::to_string(&seyk_core::ResourceLocator::WholeResource).unwrap();
        let mut resource=tx.prepare("INSERT INTO resources(profile,path,modality,size_bytes,modified_ns,content_hash,version) VALUES('p',?1,'text',?2,1,?3,?4)").unwrap();
        let mut segment=tx.prepare("INSERT INTO segments(id,profile,path,ordinal,locator,input_hash) VALUES(?1,'p',?2,0,?3,?4)").unwrap();
        let mut insert = tx.prepare("INSERT INTO segments_fts_p_70(rowid,name,path,heading,body) VALUES(?1,'note',?2,'heading',?3)").unwrap();
        let first = format!(
            "syntheticprivatequartzsentinel {}",
            words
                .split_whitespace()
                .take(199)
                .collect::<Vec<_>>()
                .join(" ")
        );
        for i in 1..=100_000 {
            let uri = format!("mem://synthetic/note{i:06}");
            let key = std::iter::once(0u8).chain(uri.bytes()).collect::<Vec<_>>();
            let hash = format!("hash{i:06}");
            let body = if i == 1 { &first } else { &words };
            resource
                .execute(params![key, body.len() as i64, hash, version])
                .unwrap();
            segment.execute(params![i, key, locator, hash]).unwrap();
            insert
                .execute(params![i, format!("synthetic/note{i:06}"), body])
                .unwrap();
        }
    }
    tx.commit().unwrap();
    db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    // One changed document in a 100k-segment profile, as in a watcher save.
    db.execute_batch("BEGIN; DELETE FROM segments_fts_p_70 WHERE rowid=1;")
        .unwrap();
    db.execute("INSERT INTO segments_fts_p_70(rowid,name,path,heading,body) VALUES(1,'note','synthetic/note000001','heading',?1)",[&words]).unwrap();
    db.execute_batch("UPDATE resources SET content_hash='changed' WHERE content_hash='hash000001'; UPDATE segments SET input_hash='changed' WHERE id=1; INSERT INTO meta VALUES('fts_scrub_pending:p','measurement'); INSERT INTO meta VALUES('wal_scrub_pending','measurement'); COMMIT;").unwrap();
    eprintln!(
        "SCRUB_COST SQLite {}: starting optimize + checked TRUNCATE on 100000 FTS segments x 200 synthetic body tokens (20000000 body-token occurrences), 1 changed document, 100000 resource records / no vectors; {} DB bytes before scrub; WAL/NORMAL; production FTS schema; temp files only",
        rusqlite::version(),
        fs::metadata(&path).unwrap().len()
    );
    let start = Instant::now();
    let report = cache.scrub().unwrap();
    let elapsed = start.elapsed();
    assert_eq!(report.profiles_scrubbed, 1);
    assert!(!report.pending);
    assert!(!cache.stats().unwrap().scrub_pending);
    let rows: i64 = db
        .query_row("SELECT count(*) FROM segments_fts_p_70", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 100000);
    let stats = cache.stats().unwrap();
    assert_eq!(stats.profiles[0].resources, 100000);
    assert_eq!(stats.profiles[0].segments, 100000);
    for file in [&path, &path.with_extension("sqlite3-wal")] {
        let bytes = fs::read(file).unwrap();
        for term in [
            "syntheticprivatequartzsentinel",
            "privatequartzsentinel",
            "quartzsentinel",
        ] {
            assert!(
                !bytes.windows(term.len()).any(|w| w == term.as_bytes()),
                "100k profile scrub {}: {term}",
                file.display()
            );
        }
    }
    eprintln!(
        "SCRUB_COST optimize + checked TRUNCATE on 100000 FTS segments x 200 synthetic body tokens (20000000 occurrences), 1 changed document, no vectors: {elapsed:?}; {} DB bytes / {} WAL bytes afterwards",
        fs::metadata(&path).unwrap().len(),
        fs::metadata(path.with_extension("sqlite3-wal"))
            .unwrap()
            .len()
    );
}
