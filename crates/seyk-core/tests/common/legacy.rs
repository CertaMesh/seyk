//! Genuine pre-v6 layouts, seeded from public synthetic fixtures.
use rusqlite::Connection;

pub fn table(profile: &str) -> String {
    format!(
        "segments_fts_p_{}",
        profile
            .bytes()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

pub fn downgrade(conn: &Connection, version: &str) {
    conn.execute_batch(
        "PRAGMA foreign_keys=OFF; BEGIN; DROP TABLE IF EXISTS image_codes;
        CREATE VIRTUAL TABLE segments_fts USING fts5(name,path,heading,body,
            content='',contentless_delete=1,tokenize='unicode61 remove_diacritics 2');",
    )
    .unwrap();
    let profiles = conn
        .prepare("SELECT id FROM profiles")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    for profile in profiles {
        let table = table(&profile);
        conn.execute_batch(&format!(
            "INSERT INTO segments_fts(rowid,name,path,heading,body)
            SELECT rowid,name,path,heading,body FROM {table}; DROP TABLE {table};"
        ))
        .unwrap();
    }
    conn.execute_batch(
        "CREATE TABLE legacy_segments (
        profile TEXT NOT NULL,path BLOB NOT NULL,ordinal INTEGER NOT NULL,
        locator TEXT NOT NULL,heading TEXT,input_hash TEXT NOT NULL,
        PRIMARY KEY(profile,path,ordinal),
        FOREIGN KEY(profile,path) REFERENCES resources(profile,path) ON DELETE CASCADE);
        INSERT INTO legacy_segments(rowid,profile,path,ordinal,locator,heading,input_hash)
            SELECT rowid,profile,path,ordinal,locator,heading,input_hash FROM segments;
        DROP TABLE segments; ALTER TABLE legacy_segments RENAME TO segments;
        CREATE INDEX segments_by_input ON segments(profile,input_hash);",
    )
    .unwrap();
    if matches!(version, "1" | "2" | "3") {
        conn.execute_batch(
            "ALTER TABLE resources DROP COLUMN version;
            ALTER TABLE resources DROP COLUMN descriptor_hash;",
        )
        .unwrap();
    }
    if version != "5" {
        conn.execute_batch("ALTER TABLE resources DROP COLUMN media_type;")
            .unwrap();
    }
    if version == "1" {
        conn.execute_batch("DROP TABLE segments_fts;").unwrap();
    }
    conn.execute(
        "UPDATE meta SET value=?1 WHERE key='schema_version'",
        [version],
    )
    .unwrap();
    conn.execute_batch("COMMIT; PRAGMA foreign_keys=ON;")
        .unwrap();
}

/// All full-detail occurrences, independent of the FTS storage layout.
pub fn postings(conn: &Connection) -> Vec<String> {
    let tables = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE
        (name='segments_fts' OR name GLOB 'segments_fts_p_*') AND sql LIKE 'CREATE VIRTUAL TABLE%'",
        )
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    let mut result = Vec::new();
    for table in tables {
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE temp.fixture_vocab USING fts5vocab(main,{table},instance)"
        ))
        .unwrap();
        result.extend(
            conn.prepare(
                "SELECT quote(term)||':'||doc||':'||col||':'||offset FROM temp.fixture_vocab",
            )
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap(),
        );
        conn.execute_batch("DROP TABLE temp.fixture_vocab").unwrap();
    }
    result.sort();
    result
}
