//! Derived Photo Anchor rows. All writes happen during an opted-in sync.
use super::*;
use seyk_embed::quant::PhotoAnchorQ4;

/// Pins the codec format, upstream revision, manifest and packing/rounding rules.
pub const IMAGE_CODEC_ID: &str = "photo-anchor-q4:v1:be9e066984345c92424c86bfcce3b17f3e3e8484:4dbde655fdd44a1feb4b010685fe065ffbe9c1f996331b4f4a40c9a5b2580c1e:hi-nibble:ties-even";

pub(super) const CREATE: &str = "CREATE TABLE IF NOT EXISTS image_codes (
    profile TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    codec_id TEXT NOT NULL,
    codes BLOB NOT NULL,
    PRIMARY KEY(profile,input_hash),
    FOREIGN KEY(profile,input_hash) REFERENCES embeddings(profile,input_hash) ON DELETE CASCADE
);";

/// Search representation for one segment. Only standalone images use codes.
#[derive(Debug, Clone)]
pub enum ScanData {
    Exact(Vec<f32>),
    PhotoAnchorQ4(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct ScanVector {
    pub resource: ResourceId,
    pub ordinal: usize,
    pub segment: CachedSegment,
    pub data: ScanData,
}

pub(crate) fn check_profile(profile: &CacheProfile) -> Result<()> {
    let model = &profile.model;
    if model.model_id != "google/embeddinggemma-2"
        || !matches!(model.dimension.get(), 256 | 768)
        || model.normalization != "mean-pool;truncate;l2"
        || !model.variant.starts_with("onnx-")
    {
        return Err(Error::Config("q4 image scan requires EmbeddingGemma 2, d256/d768 and mean-pool;truncate;l2 normalization".into()));
    }
    Ok(())
}

impl SqliteCache {
    pub(super) fn scan_rows(
        &self,
        profile: &str,
        paths: &HashSet<ResourceId>,
    ) -> Result<Vec<ScanVector>> {
        let dimension: usize = self.conn.query_row(
            "SELECT dimension FROM profiles WHERE id=?1",
            [profile],
            |r| r.get::<_, u32>(0).map(|v| v as usize),
        )?;
        let whole = serde_json::to_string(&ResourceLocator::WholeResource).map_err(json_err)?;
        let mut stmt = self.conn.prepare_cached(
            "SELECT s.path,s.ordinal,s.locator,s.heading,s.input_hash,
                CASE WHEN r.modality='image' AND s.locator=?3 AND c.codec_id=?2 THEN c.codes END,
                CASE WHEN r.modality='image' AND s.locator=?3 AND c.codec_id=?2 THEN NULL ELSE e.vector END
             FROM segments s JOIN resources r ON r.profile=s.profile AND r.path=s.path
             JOIN embeddings e ON e.profile=s.profile AND e.input_hash=s.input_hash
             LEFT JOIN image_codes c ON c.profile=s.profile AND c.input_hash=s.input_hash
             WHERE s.profile=?1 ORDER BY s.path,s.ordinal")?;
        let mut rows = stmt.query(params![profile, IMAGE_CODEC_ID, whole])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let resource = key_resource(r.get(0)?)?;
            if !paths.contains(&resource) {
                continue;
            }
            let codes: Option<Vec<u8>> = r.get(5)?;
            let data = match codes {
                Some(codes) => {
                    if !matches!(dimension, 256 | 768) || codes.len() != dimension / 2 {
                        return Err(Error::Config(
                            "corrupt current-codec image_codes row".into(),
                        ));
                    }
                    ScanData::PhotoAnchorQ4(codes)
                }
                None => {
                    let vector = from_bytes(&r.get::<_, Vec<u8>>(6)?)
                        .map_err(|_| Error::Config("corrupt f32 vector bytes".into()))?;
                    if vector.len() != dimension || vector.iter().any(|v| !v.is_finite()) {
                        return Err(Error::Config(
                            "corrupt f32 vector dimension or values".into(),
                        ));
                    }
                    ScanData::Exact(vector)
                }
            };
            out.push(ScanVector {
                resource,
                ordinal: r.get::<_, u32>(1)? as usize,
                segment: CachedSegment {
                    locator: serde_json::from_str(&r.get::<_, String>(2)?).map_err(json_err)?,
                    heading: r.get(3)?,
                    input_hash: r.get(4)?,
                },
                data,
            });
        }
        Ok(out)
    }

    pub(super) fn backfill_codes(
        &mut self,
        profile: &str,
        paths: &HashSet<ResourceId>,
    ) -> Result<()> {
        let definition: String =
            self.conn
                .query_row("SELECT profile FROM profiles WHERE id=?1", [profile], |r| {
                    r.get(0)
                })?;
        let definition: CacheProfile = serde_json::from_str(&definition).map_err(json_err)?;
        check_profile(&definition)?;
        let codec = PhotoAnchorQ4::new(definition.model.dimension.get())
            .map_err(|e| Error::Config(e.to_string()))?;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut items = HashMap::new();
        {
            let whole = serde_json::to_string(&ResourceLocator::WholeResource).map_err(json_err)?;
            let mut stmt=tx.prepare("SELECT s.path,s.input_hash,e.vector,c.codec_id,c.codes FROM segments s JOIN resources r ON r.profile=s.profile AND r.path=s.path JOIN embeddings e ON e.profile=s.profile AND e.input_hash=s.input_hash LEFT JOIN image_codes c ON c.profile=s.profile AND c.input_hash=s.input_hash WHERE s.profile=?1 AND r.modality='image' AND s.locator=?2")?;
            let mut rows = stmt.query(params![profile, whole])?;
            while let Some(r) = rows.next()? {
                if !paths.contains(&key_resource(r.get(0)?)?) {
                    continue;
                }
                let current: Option<String> = r.get(3)?;
                if current.as_deref() == Some(IMAGE_CODEC_ID) {
                    let codes: Vec<u8> = r.get(4)?;
                    if codes.len() != codec.dim() / 2 {
                        return Err(Error::Config(
                            "corrupt current-codec image_codes row".into(),
                        ));
                    }
                    continue;
                }
                let hash: String = r.get(1)?;
                if items.contains_key(&hash) {
                    continue;
                }
                let vector = from_bytes(&r.get::<_, Vec<u8>>(2)?)
                    .map_err(|_| Error::Config("corrupt f32 vector bytes".into()))?;
                // Deserialize without truncating or renormalizing the retained f32 values.
                let embedding: crate::Embedding =
                    serde_json::from_value(serde_json::to_value(vector).map_err(json_err)?)
                        .map_err(json_err)?;
                let codes = codec
                    .encode(&embedding)
                    .map_err(|e| Error::Config(e.to_string()))?;
                items.insert(hash, codes);
            }
        }
        for (hash, codes) in &items {
            tx.execute("INSERT OR REPLACE INTO image_codes(profile,input_hash,codec_id,codes) VALUES(?1,?2,?3,?4)",params![profile,hash,IMAGE_CODEC_ID,codes])?;
        }
        let generation = if items.is_empty() {
            None
        } else {
            Some(mark_cleanup(&tx, None)?)
        };
        tx.commit()?;
        self.record_write(!items.is_empty(), generation);
        Ok(())
    }

    pub(super) fn selected_vectors(
        &self,
        profile: &str,
        hashes: &HashSet<String>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let dimension: usize = self.conn.query_row(
            "SELECT dimension FROM profiles WHERE id=?1",
            [profile],
            |r| r.get::<_, u32>(0).map(|v| v as usize),
        )?;
        let mut stmt = self
            .conn
            .prepare_cached("SELECT vector FROM embeddings WHERE profile=?1 AND input_hash=?2")?;
        let mut out = HashMap::with_capacity(hashes.len());
        for hash in hashes {
            let bytes: Vec<u8> = stmt.query_row(params![profile, hash], |r| r.get(0))?;
            let values = from_bytes(&bytes)
                .map_err(|_| Error::Config("corrupt selected f32 vector".into()))?;
            if values.len() != dimension || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Config("corrupt selected f32 vector".into()));
            }
            out.insert(hash.clone(), values);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dimension, Embedding};

    fn fixture(path: &Path) -> (SqliteCache, String, HashSet<ResourceId>, Vec<f32>) {
        let mut cache = SqliteCache::open(path).unwrap();
        let profile = CacheProfile {
            model: ModelIdentity {
                model_id: "google/embeddinggemma-2".into(),
                revision: "synthetic".into(),
                variant: "onnx-q8;text-literals-v1".into(),
                dimension: Dimension::D256,
                normalization: "mean-pool;truncate;l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            },
            segmenter_version: "test".into(),
        };
        let pid = profile.id();
        cache.register_profile(&pid, &profile).unwrap();
        let vector = Embedding::from_raw(vec![1.0; 256], Dimension::D256)
            .unwrap()
            .as_slice()
            .to_vec();
        cache
            .put_embeddings(&pid, &[("shared".into(), vector.clone())])
            .unwrap();
        let mut paths = HashSet::new();
        for name in ["first", "second"] {
            let id: ResourceId = format!("test://fixture/{name}").parse().unwrap();
            let metadata = ResourceMetadata {
                modality: Modality::Image,
                media_type: Some("image/jpeg".into()),
                size_bytes: 1,
                modified_ns: None,
                content_hash: "hash".into(),
            };
            let segment = CachedSegment {
                locator: ResourceLocator::WholeResource,
                heading: None,
                input_hash: "shared".into(),
            };
            cache
                .put_resource(
                    &pid,
                    &id,
                    &ResourceWrite {
                        metadata: &metadata,
                        version: &ResourceVersion::unknown(),
                        descriptor_hash: Some("descriptor"),
                        segments: vec![SegmentWrite {
                            segment: &segment,
                            lexical_body: "",
                        }],
                    },
                )
                .unwrap();
            paths.insert(id);
        }
        (cache, pid, paths, vector)
    }

    #[test]
    fn codes_lifecycle_preserves_f32_and_cached_only_reads_do_not_backfill() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("cache.sqlite");
        let (mut cache, pid, paths, vector) = fixture(&path);
        let original = to_bytes(&vector);
        assert!(
            cache
                .image_scan_rows(&pid, &paths)
                .unwrap()
                .iter()
                .all(|r| matches!(r.data, ScanData::Exact(_)))
        );
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 0);
        cache.sync_image_codes(&pid, &paths).unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 1);
        assert!(
            cache
                .image_scan_rows(&pid, &paths)
                .unwrap()
                .iter()
                .all(|r| matches!(r.data, ScanData::PhotoAnchorQ4(_)))
        );
        cache
            .put_embeddings(&pid, &[("shared".into(), vector.clone())])
            .unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 1);
        let selected = cache
            .selected_image_vectors(&pid, &HashSet::from(["shared".into()]))
            .unwrap();
        assert_eq!(to_bytes(&selected["shared"]), original);
        let one = paths.iter().next().unwrap().clone();
        cache.remove_resources(&pid, &[one]).unwrap();
        cache.prune_orphans(&pid).unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 1);
        cache
            .put_embeddings(
                &pid,
                &[("shared".into(), vector.iter().map(|v| -v).collect())],
            )
            .unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 0);
        cache.sync_image_codes(&pid, &paths).unwrap();
        let code: Vec<u8> = cache
            .conn
            .query_row("SELECT codes FROM image_codes", [], |r| r.get(0))
            .unwrap();
        drop(cache);
        let mut cache = SqliteCache::open(&path).unwrap();
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 1);
        cache.clear().unwrap();
        drop(cache);
        for suffix in ["", "-wal"] {
            if let Ok(bytes) = std::fs::read(format!("{}{suffix}", path.display())) {
                assert!(!bytes.windows(code.len()).any(|w| w == code));
            }
        }
    }

    #[test]
    fn v6_migration_keeps_vectors_segments_descriptors_and_postings() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("cache.sqlite");
        let (cache, pid, paths, vector) = fixture(&path);
        cache
            .conn
            .execute_batch(
                "DROP TABLE image_codes; UPDATE meta SET value='6' WHERE key='schema_version';",
            )
            .unwrap();
        let before: Vec<(i64, String)> = cache
            .conn
            .prepare("SELECT id,locator FROM segments ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        drop(cache);
        let cache = SqliteCache::open(&path).unwrap();
        let after: Vec<(i64, String)> = cache
            .conn
            .prepare("SELECT id,locator FROM segments ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(before, after);
        assert_eq!(cache.vectors_for(&pid, &paths).unwrap()[0].vector, vector);
        for id in paths {
            assert_eq!(
                cache
                    .resource(&pid, &id)
                    .unwrap()
                    .unwrap()
                    .descriptor_hash
                    .as_deref(),
                Some("descriptor")
            );
        }
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 0);
    }

    #[test]
    fn unknown_codec_falls_back_but_malformed_current_codec_fails() {
        let temp = tempfile::tempdir().unwrap();
        let (mut cache, pid, paths, _) = fixture(&temp.path().join("cache.sqlite"));
        cache.sync_image_codes(&pid, &paths).unwrap();
        cache
            .conn
            .execute("UPDATE image_codes SET codec_id='future',codes=x'01'", [])
            .unwrap();
        assert!(matches!(
            cache.image_scan_rows(&pid, &paths).unwrap()[0].data,
            ScanData::Exact(_)
        ));
        cache.sync_image_codes(&pid, &paths).unwrap();
        cache
            .conn
            .execute("UPDATE image_codes SET codes=x'01'", [])
            .unwrap();
        assert!(cache.image_scan_rows(&pid, &paths).is_err());
        assert!(cache.sync_image_codes(&pid, &paths).is_err());
    }
}
