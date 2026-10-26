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
        || !matches!(
            model.variant.as_str(),
            "onnx-q8" | "onnx-fp32" | "onnx-q8;text-literals-v1" | "onnx-fp32;text-literals-v1"
        )
    {
        return Err(Error::Config("q4 image scan requires EmbeddingGemma 2, d256/d768 and mean-pool;truncate;l2 normalization".into()));
    }
    Ok(())
}

impl SqliteCache {
    fn image_dimension(&self, id: &str) -> Result<usize> {
        let (json, dimension): (String, u32) = self.conn.query_row(
            "SELECT profile,dimension FROM profiles WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let profile: CacheProfile = serde_json::from_str(&json).map_err(json_err)?;
        check_profile(&profile)?;
        if profile.id() != id || profile.model.dimension.get() != dimension as usize {
            return Err(Error::Config(
                "corrupt image profile identifier or dimension".into(),
            ));
        }
        Ok(dimension as usize)
    }
    pub(super) fn scan_rows(
        &self,
        profile: &str,
        paths: &HashSet<ResourceId>,
    ) -> Result<Vec<ScanVector>> {
        let dimension = self.image_dimension(profile)?;
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
        let codec = PhotoAnchorQ4::new(self.image_dimension(profile)?)
            .map_err(|e| Error::Config(e.to_string()))?;
        let tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut items = HashMap::new();
        {
            let whole = serde_json::to_string(&ResourceLocator::WholeResource).map_err(json_err)?;
            let mut stmt=tx.prepare("SELECT s.path,s.input_hash,CASE WHEN c.codec_id=?3 THEN NULL ELSE e.vector END,c.codec_id,c.codes FROM segments s JOIN resources r ON r.profile=s.profile AND r.path=s.path JOIN embeddings e ON e.profile=s.profile AND e.input_hash=s.input_hash LEFT JOIN image_codes c ON c.profile=s.profile AND c.input_hash=s.input_hash WHERE s.profile=?1 AND r.modality='image' AND s.locator=?2")?;
            let mut rows = stmt.query(params![profile, whole, IMAGE_CODEC_ID])?;
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
                let embedding = crate::Embedding::from_preserved(vector)
                    .map_err(|e| Error::Config(e.to_string()))?;
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
        let dimension = self.image_dimension(profile)?;
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
        fixture_with_dimension(path, Dimension::D256)
    }

    fn fixture_with_dimension(
        path: &Path,
        dimension: Dimension,
    ) -> (SqliteCache, String, HashSet<ResourceId>, Vec<f32>) {
        let mut cache = SqliteCache::open(path).unwrap();
        let profile = CacheProfile {
            model: ModelIdentity {
                model_id: "google/embeddinggemma-2".into(),
                revision: "synthetic".into(),
                variant: "onnx-q8;text-literals-v1".into(),
                dimension,
                normalization: "mean-pool;truncate;l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            },
            segmenter_version: "test".into(),
        };
        let pid = profile.id();
        cache.register_profile(&pid, &profile).unwrap();
        let vector = Embedding::from_raw(vec![1.0; dimension.get()], dimension)
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
    fn backfilled_codes_match_serde_restoration_byte_for_byte() {
        for dim in [256, 768] {
            let temp = tempfile::tempdir().unwrap();
            let (mut cache, pid, paths, _) = fixture_with_dimension(
                &temp.path().join("cache.sqlite"),
                Dimension::try_from(dim).unwrap(),
            );
            for phase in [0.0, 0.37, 1.5] {
                let mut vector = Embedding::from_raw(
                    (0..dim)
                        .map(|i| {
                            if i < 2 {
                                0.0
                            } else {
                                (i as f32 * 0.11 + phase).sin()
                            }
                        })
                        .collect(),
                    Dimension::try_from(dim).unwrap(),
                )
                .unwrap()
                .into_vec();
                vector[0] = -0.0;
                vector[1] = f32::from_bits(1);
                let legacy: Embedding =
                    serde_json::from_value(serde_json::to_value(&vector).unwrap()).unwrap();
                let expected = PhotoAnchorQ4::new(dim).unwrap().encode(&legacy).unwrap();
                cache
                    .put_embeddings(&pid, &[("shared".into(), vector.clone())])
                    .unwrap();
                cache.sync_image_codes(&pid, &paths).unwrap();
                let actual: Vec<u8> = cache
                    .conn
                    .query_row("SELECT codes FROM image_codes", [], |r| r.get(0))
                    .unwrap();
                assert_eq!(actual, expected, "dim={dim}, phase={phase}");
                let retained = cache
                    .selected_image_vectors(&pid, &HashSet::from(["shared".into()]))
                    .unwrap();
                assert_eq!(to_bytes(&retained["shared"]), to_bytes(&vector));
            }
        }
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
    #[test]
    fn q4_profile_corruption_is_rejected_on_reads_and_writes() {
        let temp = tempfile::tempdir().unwrap();
        let (mut cache, pid, paths, _) = fixture(&temp.path().join("cache.sqlite"));
        cache
            .conn
            .execute("UPDATE profiles SET dimension=768 WHERE id=?1", [&pid])
            .unwrap();
        assert!(cache.image_scan_rows(&pid, &paths).is_err());
        assert!(cache.sync_image_codes(&pid, &paths).is_err());
        assert!(
            cache
                .selected_image_vectors(&pid, &HashSet::from(["shared".into()]))
                .is_err()
        );
    }

    #[test]
    fn pdf_pages_with_shared_image_inputs_keep_exact_f32() {
        let temp = tempfile::tempdir().unwrap();
        let (mut cache, pid, mut paths, _) = fixture(&temp.path().join("cache.sqlite"));
        let page: ResourceId = "test://fixture/document.pdf".parse().unwrap();
        let metadata = ResourceMetadata {
            modality: Modality::Document,
            media_type: Some("application/pdf".into()),
            size_bytes: 1,
            modified_ns: None,
            content_hash: "hash".into(),
        };
        let segment = CachedSegment {
            locator: ResourceLocator::Page { page: 1 },
            heading: None,
            input_hash: "shared".into(),
        };
        cache
            .put_resource(
                &pid,
                &page,
                &ResourceWrite {
                    metadata: &metadata,
                    version: &ResourceVersion::unknown(),
                    descriptor_hash: Some("pdf"),
                    segments: vec![SegmentWrite {
                        segment: &segment,
                        lexical_body: "",
                    }],
                },
            )
            .unwrap();
        paths.insert(page.clone());
        cache.sync_image_codes(&pid, &paths).unwrap();
        let rows = cache.image_scan_rows(&pid, &paths).unwrap();
        assert!(
            rows.iter()
                .filter(|r| r.resource == page)
                .all(|r| matches!(r.data, ScanData::Exact(_)))
        );
        assert_eq!(
            rows.iter()
                .filter(|r| matches!(r.data, ScanData::PhotoAnchorQ4(_)))
                .count(),
            2
        );
        assert_eq!(cache.stats().unwrap().profiles[0].image_codes, 1);
    }

    #[test]
    fn removing_the_final_segment_reference_prunes_derived_codes() {
        let temp = tempfile::tempdir().unwrap();
        let (mut cache, pid, paths, _) = fixture(&temp.path().join("cache.sqlite"));
        cache.sync_image_codes(&pid, &paths).unwrap();
        cache
            .remove_resources(&pid, &paths.into_iter().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(cache.prune_orphans(&pid).unwrap(), 1);
        let stats = cache.stats().unwrap();
        assert_eq!(stats.profiles[0].embeddings, 0);
        assert_eq!(stats.profiles[0].image_codes, 0);
    }
}
