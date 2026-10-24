//! Verified, filtering import into a fresh cache. No extraction or model access.
use super::*;
use crate::discovery::{ResourceKind, modified_ns};
use crate::provider::{FsProvider, ProvidedResource};
use std::collections::BTreeMap;
use std::io::{BufRead, Read};

/// One explicit, one-to-one local file mapping in JSONL format version 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheImportMapping {
    pub version: u32,
    pub old: ResourceId,
    pub new: ResourceId,
    /// Caller-supplied full-file BLAKE3, 64 lowercase hexadecimal characters.
    pub content_hash: String,
}

impl CacheImportMapping {
    /// Parse nonblank JSONL lines. Diagnostics contain line numbers, never input data.
    pub fn read_jsonl(reader: impl BufRead) -> Result<Vec<Self>> {
        let mut mappings = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line.map_err(|_| invalid("cannot read mapping file"))?;
            if line.trim().is_empty() {
                continue;
            }
            let mapping = serde_json::from_str(&line)
                .map_err(|_| invalid(&format!("invalid mapping on line {}", index + 1)))?;
            mappings.push(mapping);
        }
        validate_mappings(&mappings)?;
        Ok(mappings)
    }
}

/// Counts only: no source excerpts, terms, paths or vector bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheImportReport {
    pub resources: u64,
    pub segments: u64,
    pub vectors: u64,
    pub fts_rows: u64,
    /// All five reasons are always listed, including zero counts: hash_mismatch,
    /// missing_hash, preparation_changed, unavailable_or_unstable and unmapped.
    pub drops: BTreeMap<String, u64>,
    pub unmatched_mappings: u64,
}

impl Default for CacheImportReport {
    fn default() -> Self {
        Self {
            resources: 0,
            segments: 0,
            vectors: 0,
            fts_rows: 0,
            unmatched_mappings: 0,
            drops: [
                "hash_mismatch",
                "missing_hash",
                "preparation_changed",
                "unavailable_or_unstable",
                "unmapped",
            ]
            .into_iter()
            .map(|reason| (reason.into(), 0))
            .collect(),
        }
    }
}

impl CacheImportReport {
    fn drop_resource(&mut self, reason: &str) {
        *self.drops.entry(reason.into()).or_default() += 1;
    }
}

fn invalid(message: &str) -> Error {
    Error::Config(format!("cache import: {message}"))
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_mappings(mappings: &[CacheImportMapping]) -> Result<()> {
    let mut old = HashSet::new();
    let mut new = HashSet::new();
    for mapping in mappings {
        if mapping.version != 1 || !valid_hash(&mapping.content_hash) {
            return Err(invalid(
                "mapping requires version 1 and a lowercase BLAKE3 hash",
            ));
        }
        if mapping.old.to_file_path().is_none() || mapping.new.to_file_path().is_none() {
            return Err(invalid("mapping requires absolute local file URIs"));
        }
        if !old.insert(&mapping.old) || !new.insert(&mapping.new) {
            return Err(invalid("duplicate old or new mapping identity"));
        }
    }
    Ok(())
}

fn reject_alias(source: &Path, destination: &Path) -> Result<()> {
    let src = std::fs::canonicalize(source).map_err(|e| Error::io(source, e))?;
    if destination.exists() {
        let dst = std::fs::canonicalize(destination).map_err(|e| Error::io(destination, e))?;
        let mut alias = src == dst;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let a = std::fs::metadata(source).map_err(|e| Error::io(source, e))?;
            let b = std::fs::metadata(destination).map_err(|e| Error::io(destination, e))?;
            alias |= (a.dev(), a.ino()) == (b.dev(), b.ino());
        }
        if alias {
            return Err(invalid("source and destination identify the same file"));
        }
    }
    Ok(())
}

fn source_snapshot(source: &Path) -> Result<Connection> {
    let absolute = std::fs::canonicalize(source).map_err(|e| Error::io(source, e))?;
    let mut uri = url::Url::from_file_path(absolute).map_err(|_| invalid("invalid source path"))?;
    uri.set_query(Some("mode=ro"));
    let conn = Connection::open_with_flags(
        uri.as_str(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.execute_batch("BEGIN")?;
    let has_meta: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='meta')",
        [],
        |r| r.get(0),
    )?;
    if !has_meta {
        return Err(invalid("source requires schema v6"));
    }
    let version: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='schema_version'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if version.as_deref() != Some(SCHEMA_VERSION) {
        return Err(invalid(
            "source requires schema v6; open older caches once with this seyk first",
        ));
    }
    Ok(conn)
}

fn read_profile(conn: &Connection, id: &str) -> Result<CacheProfile> {
    let (json, dimension): (String, i64) = conn
        .query_row(
            "SELECT profile,dimension FROM profiles WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?
        .ok_or_else(|| invalid("profile not found in source"))?;
    let profile: CacheProfile =
        serde_json::from_str(&json).map_err(|_| invalid("invalid profile definition"))?;
    let stored: serde_json::Value =
        serde_json::from_str(&json).map_err(|_| invalid("invalid profile definition"))?;
    if stored != serde_json::to_value(&profile).map_err(json_err)?
        || profile.id() != id
        || dimension != profile.model.dimension.get() as i64
    {
        return Err(invalid("profile identifier or dimension mismatch"));
    }
    Ok(profile)
}

impl SqliteCache {
    /// Select a validated profile without opening the source as a writable cache.
    pub fn import_profile(source: &Path, id: &str) -> Result<CacheProfile> {
        read_profile(&source_snapshot(source)?, id)
    }

    /// Seed an absent or completely empty v6 cache from verified local files.
    /// Only schema v6 sources are supported. Errors roll back schema and data;
    /// verification failures drop individual resources and are reported as counts.
    /// PDF/audio preparation compatibility is also checked by the runtime on reuse.
    pub fn import_from(
        source: &Path,
        destination: &Path,
        profile: &CacheProfile,
        mappings: &[CacheImportMapping],
    ) -> Result<CacheImportReport> {
        Self::import_inner(source, destination, profile, mappings, |_| Ok(()))
    }

    fn import_inner(
        source: &Path,
        destination: &Path,
        profile: &CacheProfile,
        mappings: &[CacheImportMapping],
        mut checkpoint: impl FnMut(&str) -> Result<()>,
    ) -> Result<CacheImportReport> {
        validate_mappings(mappings)?;
        reject_alias(source, destination)?;
        let src = source_snapshot(source)?;
        let id = profile.id();
        if read_profile(&src, &id)? != *profile {
            return Err(invalid("profile definition mismatch"));
        }
        let mut dst = Self::open_internal(
            destination.to_path_buf(),
            crate::cache_guards::prepare,
            false,
        )?;
        let tx = dst
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let has_meta: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='meta')",
            [],
            |r| r.get(0),
        )?;
        if has_meta {
            let version: Option<String> = tx
                .query_row(
                    "SELECT value FROM meta WHERE key='schema_version'",
                    [],
                    |r| r.get(0),
                )
                .optional()?;
            if version.as_deref() != Some(SCHEMA_VERSION) {
                return Err(invalid("destination must be fresh or empty schema v6"));
            }
        }
        let extra: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'
             AND NOT (type='table' AND name IN ('meta','profiles','resources','segments','embeddings'))
             AND NOT (type='index' AND name='segments_by_input' AND tbl_name='segments'))",
            [], |r| r.get(0))?;
        if extra {
            return Err(invalid("destination contains foreign schema objects"));
        }
        Self::initialize_schema(&tx)?;
        for table in ["profiles", "resources", "segments", "embeddings"] {
            let count: i64 =
                tx.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))?;
            if count != 0 {
                return Err(invalid("destination is populated"));
            }
        }
        let metadata_count: i64 = tx.query_row(
            "SELECT count(*) FROM meta WHERE key != 'schema_version'",
            [],
            |r| r.get(0),
        )?;
        if metadata_count != 0 {
            return Err(invalid("destination has pending metadata"));
        }
        tx.execute(
            "INSERT INTO profiles VALUES(?1,?2,?3)",
            params![
                id,
                serde_json::to_string(profile).map_err(json_err)?,
                profile.model.dimension.get() as i64
            ],
        )?;
        create_profile_fts(&tx, &id)?;
        let mut report = CacheImportReport::default();
        let source_count: i64 = src.query_row(
            "SELECT count(*) FROM resources WHERE profile=?1",
            [&id],
            |r| r.get(0),
        )?;
        let mut matched = 0;
        for mapping in mappings {
            let old = resource_key(&mapping.old);
            let row: Option<(String, Option<String>, String, Option<String>)> = src.query_row(
                "SELECT content_hash,descriptor_hash,modality,media_type FROM resources WHERE profile=?1 AND path=?2",
                params![id, old], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?;
            let Some((hash, descriptor, modality, media_type)) = row else {
                report.unmatched_mappings += 1;
                continue;
            };
            matched += 1;
            if !valid_hash(&hash) {
                report.drop_resource("missing_hash");
                continue;
            }
            let Some(descriptor) = descriptor else {
                report.drop_resource("preparation_changed");
                continue;
            };
            validate_resource_type(&modality, media_type.as_deref())?;
            let Some(new_descriptor) = path_descriptor(&mapping.new) else {
                report.drop_resource("preparation_changed");
                continue;
            };
            let Some(old_descriptor) = path_descriptor(&mapping.old) else {
                report.drop_resource("preparation_changed");
                continue;
            };
            if old_descriptor.fingerprint() != new_descriptor.fingerprint()
                || (matches!(new_descriptor.kind, ResourceKind::Text(_))
                    && descriptor != new_descriptor.fingerprint())
                || new_descriptor.kind.modality().to_string() != modality
            {
                report.drop_resource("preparation_changed");
                continue;
            }
            let verified = match verify_file(&mapping.new) {
                Ok(value) => value,
                Err(_) => {
                    report.drop_resource("unavailable_or_unstable");
                    continue;
                }
            };
            if verified.0 != hash || mapping.content_hash != hash {
                report.drop_resource("hash_mismatch");
                continue;
            }
            let key = resource_key(&mapping.new);
            let version = ResourceVersion::file(verified.1, verified.2);
            tx.execute("INSERT INTO resources(profile,path,modality,size_bytes,modified_ns,content_hash,version,descriptor_hash,media_type) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![id,key,modality,verified.1 as i64,verified.2,hash,serde_json::to_string(&version).map_err(json_err)?,descriptor,media_type])?;
            checkpoint("resource")?;
            import_segments(
                &src,
                &tx,
                &id,
                profile,
                &old,
                &mapping.new,
                verified.1,
                &mut report,
                &mut checkpoint,
            )?;
            report.resources += 1;
        }
        report
            .drops
            .insert("unmapped".into(), source_count as u64 - matched);
        let violations: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_foreign_key_check)",
            [],
            |r| r.get(0),
        )?;
        let missing: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM segments s LEFT JOIN embeddings e ON s.profile=e.profile AND s.input_hash=e.input_hash WHERE e.input_hash IS NULL)", [], |r| r.get(0))?;
        let fts_count: i64 = tx.query_row(
            &format!("SELECT count(*) FROM {}", fts_table(&id)),
            [],
            |r| r.get(0),
        )?;
        if violations || missing || fts_count as u64 != report.segments {
            return Err(invalid("destination integrity check failed"));
        }
        tx.execute_batch(&format!(
            "INSERT INTO {0}({0}) VALUES('integrity-check')",
            fts_table(&id)
        ))?;
        checkpoint("commit")?;
        tx.commit()?;
        Ok(report)
    }
}

fn validate_resource_type(modality: &str, media_type: Option<&str>) -> Result<()> {
    if matches!(
        (modality, media_type),
        ("text" | "image" | "video", None)
            | ("document", Some("application/pdf"))
            | (
                "audio",
                Some("audio/wav" | "audio/flac" | "audio/mp4" | "audio/mpeg")
            )
    ) {
        Ok(())
    } else {
        Err(invalid("invalid modality or media type"))
    }
}

fn path_descriptor(id: &ResourceId) -> Option<ProvidedResource> {
    let path = id.to_file_path()?;
    Some(ProvidedResource {
        id: id.clone(),
        kind: ResourceKind::from_path(&path)?,
        size_bytes: 0,
        modified_ns: None,
        version: ResourceVersion::unknown(),
        title: path.file_stem().map(|s| s.to_string_lossy().into_owned()),
    })
}

fn stable_metadata(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    let same =
        a.is_file() && b.is_file() && a.len() == b.len() && a.modified().ok() == b.modified().ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        same && (a.dev(), a.ino(), a.ctime(), a.ctime_nsec())
            == (b.dev(), b.ino(), b.ctime(), b.ctime_nsec())
    }
    #[cfg(not(unix))]
    {
        same
    }
}

fn verify_file(id: &ResourceId) -> Result<(String, u64, Option<i64>)> {
    verify_file_after_read(id, || {})
}

fn verify_file_after_read(
    id: &ResourceId,
    after_read: impl FnOnce(),
) -> Result<(String, u64, Option<i64>)> {
    let mut file = FsProvider::open_verified_handle(id)?;
    let path = id.to_file_path().expect("validated local mapping");
    let before = file.metadata().map_err(|e| Error::io(&path, e))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| Error::io(&path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    after_read();
    let after = file.metadata().map_err(|e| Error::io(&path, e))?;
    let current = FsProvider::open_verified_handle(id)?
        .metadata()
        .map_err(|e| Error::io(&path, e))?;
    if !stable_metadata(&before, &after) || !stable_metadata(&after, &current) {
        return Err(Error::ChangedWhileReading {
            resource: id.clone(),
        });
    }
    Ok((
        hasher.finalize().to_hex().to_string(),
        after.len(),
        modified_ns(&after),
    ))
}

#[allow(clippy::too_many_arguments)]
fn import_segments(
    src: &Connection,
    tx: &rusqlite::Transaction<'_>,
    id: &str,
    profile: &CacheProfile,
    old: &[u8],
    new: &ResourceId,
    size: u64,
    report: &mut CacheImportReport,
    checkpoint: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stmt = src.prepare("SELECT id,ordinal,locator,heading,input_hash FROM segments WHERE profile=?1 AND path=?2 ORDER BY ordinal")?;
    let mut rows = stmt.query(params![id, old])?;
    let (name, path) = new.labels();
    let mut expected_ordinal = 0;
    while let Some(row) = rows.next()? {
        let rowid: i64 = row.get(0)?;
        let ordinal: i64 = row.get(1)?;
        if ordinal != expected_ordinal {
            return Err(invalid("noncontiguous segment ordinals"));
        }
        expected_ordinal += 1;
        let locator_json: String = row.get(2)?;
        let locator: ResourceLocator =
            serde_json::from_str(&locator_json).map_err(|_| invalid("malformed locator"))?;
        let stored_locator: serde_json::Value =
            serde_json::from_str(&locator_json).map_err(|_| invalid("malformed locator"))?;
        if stored_locator != serde_json::to_value(&locator).map_err(json_err)? {
            return Err(invalid("unexpected locator fields"));
        }
        validate_locator(&locator, size)?;
        let heading: Option<String> = row.get(3)?;
        let input: String = row.get(4)?;
        if !valid_hash(&input) {
            return Err(invalid("invalid input hash"));
        }
        let vector: Vec<u8> = src.query_row(
            "SELECT vector FROM embeddings WHERE profile=?1 AND input_hash=?2",
            params![id, input],
            |r| r.get(0),
        )?;
        if vector.len() != profile.model.dimension.get() * 4
            || from_bytes(&vector).map_or(true, |values| values.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("invalid vector encoding or dimension"));
        }
        let (fts_heading, body): (String, String) = src.query_row(
            &format!("SELECT heading,body FROM {} WHERE rowid=?1", fts_table(id)),
            [rowid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        if order_free_text(&fts_heading) != fts_heading
            || order_free_text(&body) != body
            || order_free_text(heading.as_deref().unwrap_or("")) != fts_heading
        {
            return Err(invalid("inconsistent or non-order-free keyword row"));
        }
        report.vectors += tx.execute(
            "INSERT OR IGNORE INTO embeddings VALUES(?1,?2,?3)",
            params![id, input, vector],
        )? as u64;
        checkpoint("vector")?;
        tx.execute("INSERT INTO segments(profile,path,ordinal,locator,heading,input_hash) VALUES(?1,?2,?3,?4,?5,?6)",params![id,resource_key(new),ordinal,serde_json::to_string(&locator).map_err(json_err)?,heading,input])?;
        checkpoint("segment")?;
        tx.execute(
            &format!(
                "INSERT INTO {}(rowid,name,path,heading,body) VALUES(?1,?2,?3,?4,?5)",
                fts_table(id)
            ),
            params![
                tx.last_insert_rowid(),
                order_free_text(&name),
                order_free_text(&path),
                fts_heading,
                body
            ],
        )?;
        checkpoint("fts")?;
        report.segments += 1;
        report.fts_rows += 1;
    }
    Ok(())
}

fn validate_locator(locator: &ResourceLocator, size: u64) -> Result<()> {
    let valid = match locator {
        ResourceLocator::Text {
            byte_start,
            byte_end,
            line_start,
            line_end,
        } => {
            byte_start <= byte_end
                && *byte_end <= size
                && match (line_start, line_end) {
                    (Some(a), Some(b)) => *a > 0 && a <= b,
                    (None, None) => true,
                    _ => false,
                }
        }
        ResourceLocator::PageText {
            page,
            byte_start,
            byte_end,
            text_version,
            page_text_hash,
        } => {
            *page > 0
                && byte_start <= byte_end
                && text_version == crate::resource::PDF_TEXT_VERSION
                && valid_hash(page_text_hash)
        }
        ResourceLocator::Page { page } => *page > 0,
        ResourceLocator::Time { start_ms, end_ms } => start_ms < end_ms,
        ResourceLocator::ImageRegion { width, height, .. } => *width > 0 && *height > 0,
        ResourceLocator::WholeResource => true,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid("invalid locator range"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dimension, ModelIdentity};

    fn fixture(root: &Path) -> (PathBuf, PathBuf, CacheProfile, CacheImportMapping) {
        let old = root.join("old/note.txt");
        let new = root.join("new/note.txt");
        std::fs::create_dir_all(new.parent().unwrap()).unwrap();
        std::fs::write(&new, b"violet amber violet").unwrap();
        let source = root.join("source.sqlite");
        let destination = root.join("destination.sqlite");
        let profile = CacheProfile {
            model: ModelIdentity {
                model_id: "synthetic".into(),
                revision: "1".into(),
                variant: "test".into(),
                dimension: Dimension::D128,
                normalization: "l2".into(),
                query_prompt: "q".into(),
                document_prompt: "d".into(),
                max_tokens: 512,
            },
            segmenter_version: "test".into(),
        };
        let mapping = CacheImportMapping {
            version: 1,
            old: ResourceId::from_path(old).unwrap(),
            new: ResourceId::from_path(new).unwrap(),
            content_hash: blake3::hash(b"violet amber violet").to_hex().to_string(),
        };
        let mut cache = SqliteCache::open(&source).unwrap();
        cache.register_profile(&profile.id(), &profile).unwrap();
        let input_hash = blake3::hash(b"synthetic model input").to_hex().to_string();
        cache
            .put_embeddings(&profile.id(), &[(input_hash.clone(), vec![0.25; 128])])
            .unwrap();
        let segment = CachedSegment {
            locator: ResourceLocator::Text {
                byte_start: 0,
                byte_end: 19,
                line_start: Some(1),
                line_end: Some(1),
            },
            heading: None,
            input_hash,
        };
        let descriptor = path_descriptor(&mapping.old).unwrap().fingerprint();
        cache
            .put_resource(
                &profile.id(),
                &mapping.old,
                &ResourceWrite {
                    metadata: &ResourceMetadata {
                        modality: Modality::Text,
                        media_type: None,
                        size_bytes: 19,
                        modified_ns: None,
                        content_hash: mapping.content_hash.clone(),
                    },
                    version: &ResourceVersion::unknown(),
                    descriptor_hash: Some(&descriptor),
                    segments: vec![SegmentWrite {
                        segment: &segment,
                        lexical_body: "violet amber violet",
                    }],
                },
            )
            .unwrap();
        (source, destination, profile, mapping)
    }

    #[test]
    fn every_write_stage_and_precommit_failure_roll_back_schema_and_rows() {
        for initialized in [false, true] {
            for stage in ["resource", "vector", "segment", "fts", "commit"] {
                let temp = tempfile::tempdir().unwrap();
                let root = temp.path().canonicalize().unwrap();
                let (source, destination, profile, mapping) = fixture(&root);
                if initialized {
                    drop(SqliteCache::open(&destination).unwrap());
                }
                let before_source = std::fs::read(&source).unwrap();
                let fail = SqliteCache::import_inner(
                    &source,
                    &destination,
                    &profile,
                    std::slice::from_ref(&mapping),
                    |at| {
                        if at == stage {
                            Err(invalid("injected failure"))
                        } else {
                            Ok(())
                        }
                    },
                );
                assert!(fail.is_err(), "{stage}");
                assert_eq!(std::fs::read(&source).unwrap(), before_source);
                let conn = Connection::open(&destination).unwrap();
                let count: i64 = conn.query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",[],|r| r.get(0)).unwrap();
                assert_eq!(count, if initialized { 5 } else { 0 }, "{stage}");
                drop(conn);
                assert_eq!(
                    SqliteCache::import_from(&source, &destination, &profile, &[mapping])
                        .unwrap()
                        .resources,
                    1
                );
            }
        }
    }

    #[test]
    fn handle_mutation_replacement_and_disappearance_fail_verification() {
        for action in ["mutate", "replace", "delete"] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().canonicalize().unwrap();
            let (_, _, _, mapping) = fixture(&root);
            let path = mapping.new.to_file_path().unwrap();
            let result = verify_file_after_read(&mapping.new, || match action {
                "mutate" => std::fs::write(&path, b"changed bytes with different length").unwrap(),
                "replace" => {
                    let replacement = path.with_extension("replacement");
                    std::fs::write(&replacement, b"violet amber violet").unwrap();
                    std::fs::rename(replacement, &path).unwrap();
                }
                _ => std::fs::remove_file(&path).unwrap(),
            });
            assert!(result.is_err(), "{action}");
        }
    }

    #[test]
    fn imported_locator_is_reencoded_in_canonical_form() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (source, destination, profile, mapping) = fixture(&root);
        let conn = Connection::open(&source).unwrap();
        let original: String = conn
            .query_row("SELECT locator FROM segments", [], |r| r.get(0))
            .unwrap();
        let locator: ResourceLocator = serde_json::from_str(&original).unwrap();
        let pretty = serde_json::to_string_pretty(&locator).unwrap();
        assert_ne!(original, pretty);
        conn.execute("UPDATE segments SET locator=?1", [&pretty])
            .unwrap();
        SqliteCache::import_from(&source, &destination, &profile, &[mapping]).unwrap();
        let stored: String = Connection::open(destination)
            .unwrap()
            .query_row("SELECT locator FROM segments", [], |r| r.get(0))
            .unwrap();
        assert_eq!(stored, serde_json::to_string(&locator).unwrap());
        assert_ne!(stored, pretty);
    }

    #[cfg(unix)]
    #[test]
    fn same_length_mutation_with_restored_mtime_fails_verification() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (_, _, _, mapping) = fixture(&root);
        let path = mapping.new.to_file_path().unwrap();
        let before = std::fs::metadata(&path).unwrap();
        let result = verify_file_after_read(&mapping.new, || {
            std::fs::write(&path, b"amber violet violet").unwrap();
            std::fs::File::open(&path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(before.modified().unwrap()))
                .unwrap();
        });
        let after = std::fs::metadata(path).unwrap();
        assert_eq!(before.len(), after.len());
        assert_eq!(before.modified().unwrap(), after.modified().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn source_wal_snapshot_stays_consistent_during_concurrent_commit() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (source, destination, profile, mapping) = fixture(&root);
        // Keep WAL handles live while the importer and a writer hold distinct connections.
        let writer = Connection::open(&source).unwrap();
        let report = SqliteCache::import_inner(&source, &destination, &profile, &[mapping], |at| {
            if at == "resource" {
                writer.execute("UPDATE embeddings SET vector=x'00'", [])?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(report.resources, 1);
        let width: i64 = Connection::open(destination)
            .unwrap()
            .query_row("SELECT length(vector) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(width, 512);
    }

    #[test]
    fn pdf_and_audio_keep_composite_descriptors_for_runtime_validation() {
        for (extension, modality, locator) in [
            (
                "pdf",
                "document",
                ResourceLocator::PageText {
                    page: 1,
                    byte_start: 0,
                    byte_end: 3,
                    text_version: crate::resource::PDF_TEXT_VERSION.into(),
                    page_text_hash: blake3::hash(b"synthetic page").to_hex().to_string(),
                },
            ),
            (
                "wav",
                "audio",
                ResourceLocator::Time {
                    start_ms: 0,
                    end_ms: 10,
                },
            ),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path().canonicalize().unwrap();
            let (source, destination, profile, mut mapping) = fixture(&root);
            let old = mapping
                .old
                .to_file_path()
                .unwrap()
                .with_extension(extension);
            let new = mapping
                .new
                .to_file_path()
                .unwrap()
                .with_extension(extension);
            std::fs::rename(mapping.new.to_file_path().unwrap(), &new).unwrap();
            mapping.old = ResourceId::from_path(old).unwrap();
            mapping.new = ResourceId::from_path(new).unwrap();
            let conn = Connection::open(&source).unwrap();
            conn.execute_batch("BEGIN; PRAGMA defer_foreign_keys=ON;")
                .unwrap();
            conn.execute("UPDATE resources SET path=?1,modality=?2,media_type=?3,descriptor_hash='composite-runtime-identity'",params![resource_key(&mapping.old),modality,if modality == "document" { "application/pdf" } else { "audio/wav" }]).unwrap();
            conn.execute(
                "UPDATE segments SET path=?1,locator=?2",
                params![
                    resource_key(&mapping.old),
                    serde_json::to_string(&locator).unwrap()
                ],
            )
            .unwrap();
            conn.execute_batch("COMMIT").unwrap();
            assert_eq!(
                SqliteCache::import_from(
                    &source,
                    &destination,
                    &profile,
                    std::slice::from_ref(&mapping)
                )
                .unwrap()
                .resources,
                1
            );
            let resource = SqliteCache::open(destination)
                .unwrap()
                .resource(&profile.id(), &mapping.new)
                .unwrap()
                .unwrap();
            assert_eq!(
                resource.descriptor_hash.as_deref(),
                Some("composite-runtime-identity")
            );
            assert_eq!(resource.segments[0].locator, locator);
        }
    }

    #[test]
    fn verified_metadata_and_provider_token_come_from_opened_file() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (source, destination, profile, mapping) = fixture(&root);
        SqliteCache::import_from(
            &source,
            &destination,
            &profile,
            std::slice::from_ref(&mapping),
        )
        .unwrap();
        let cache = SqliteCache::open(destination).unwrap();
        let resource = cache
            .resource(&profile.id(), &mapping.new)
            .unwrap()
            .unwrap();
        let metadata = std::fs::metadata(mapping.new.to_file_path().unwrap()).unwrap();
        assert_eq!(resource.metadata.size_bytes, metadata.len());
        assert_eq!(resource.metadata.modified_ns, modified_ns(&metadata));
        assert_eq!(
            resource.version,
            ResourceVersion::file(metadata.len(), modified_ns(&metadata))
        );
    }
}
