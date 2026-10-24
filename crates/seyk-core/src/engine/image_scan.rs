use super::*;
use crate::cache::{ScanData, ScanVector};
use seyk_embed::quant::QueryLut;

/// Experimental image scan. Returned scores always use the retained exact f32 vectors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ImageScan {
    #[default]
    Exact,
    PhotoAnchorQ4,
}

impl std::str::FromStr for ImageScan {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "exact" => Ok(Self::Exact),
            "q4" => Ok(Self::PhotoAnchorQ4),
            _ => Err(Error::Config("unknown image scan: use exact or q4".into())),
        }
    }
}

pub(super) fn search(
    engine: &SearchEngine,
    rows: &[ScanVector],
    queries: &[Embedding],
    take: usize,
    min_score: f32,
) -> Result<Vec<Vec<Scored>>> {
    if take == 0 {
        return Ok(vec![Vec::new(); queries.len()]);
    }
    let coded = rows
        .iter()
        .filter(|r| matches!(r.data, ScanData::PhotoAnchorQ4(_)))
        .count();
    let depth = coded.min(take.saturating_mul(4).max(50));
    let mut shortlists = Vec::with_capacity(queries.len());
    let mut hashes = HashSet::new();
    for query in queries {
        let lut = QueryLut::new(query).map_err(|e| Error::Config(e.to_string()))?;
        let mut approx = Vec::with_capacity(coded);
        for (id, row) in rows.iter().enumerate() {
            if let ScanData::PhotoAnchorQ4(codes) = &row.data {
                approx.push(Scored {
                    id,
                    score: lut.score(codes).map_err(|e| Error::Config(e.to_string()))?,
                });
            }
        }
        approx.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        approx.truncate(depth);
        for item in &approx {
            hashes.insert(rows[item.id].segment.input_hash.clone());
        }
        shortlists.push(approx);
    }
    let exact = engine.with_cache(|c| c.selected_image_vectors(&engine.profile_id, &hashes))?;
    queries
        .iter()
        .zip(shortlists)
        .map(|(query, shortlist)| {
            let mut scores = Vec::new();
            for (id, row) in rows.iter().enumerate() {
                if let ScanData::Exact(vector) = &row.data {
                    scores.push(Scored {
                        id,
                        score: crate::embedding::dot(query.as_slice(), vector),
                    });
                }
            }
            for candidate in shortlist {
                let vector = exact
                    .get(&rows[candidate.id].segment.input_hash)
                    .ok_or_else(|| Error::Config("selected image f32 vector disappeared".into()))?;
                scores.push(Scored {
                    id: candidate.id,
                    score: crate::embedding::dot(query.as_slice(), vector),
                });
            }
            scores.retain(|s| s.score >= min_score);
            scores.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
            scores.truncate(take);
            Ok(scores)
        })
        .collect()
}
