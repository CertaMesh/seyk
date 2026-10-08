//! Nearest-neighbour search over normalised vectors.
//!
//! [`BruteForceIndex`] scores every vector. It is exact, deterministic and has
//! no build step, which suits a lazily filled cache. An approximate index
//! (HNSW or similar) can implement [`VectorIndex`] when collections get large.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::embedding::dot;
use crate::{Error, Result};

/// A scored match: `id` is the position the vector was added at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub id: usize,
    pub score: f32,
}

pub trait VectorIndex {
    fn dimension(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Add a normalised vector, returning its id.
    fn add(&mut self, vector: &[f32]) -> usize;
    /// The `limit` best matches with score >= `min_score`, best first.
    fn search(&self, query: &[f32], limit: usize, min_score: f32) -> Vec<Scored>;
}

/// Exact search by scanning a contiguous matrix.
#[derive(Debug, Clone)]
pub struct BruteForceIndex {
    dim: usize,
    data: Vec<f32>,
}

impl BruteForceIndex {
    /// Create an empty index. Returns [`Error::Config`] if `dim` is zero.
    pub fn new(dim: usize) -> Result<Self> {
        Self::with_capacity(dim, 0)
    }

    /// Reserve space for `n` vectors. Returns [`Error::Config`] if `dim` is zero.
    pub fn with_capacity(dim: usize, n: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Config(
                "vector index dimension must be nonzero".into(),
            ));
        }
        Ok(Self {
            dim,
            data: Vec::with_capacity(dim * n),
        })
    }

    /// Bytes held by the vector matrix.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

// Min-heap entry: the worst of the current top-k sits on top.
#[derive(PartialEq)]
struct Entry(Scored);

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse score so the heap top is the lowest score; ties keep lower ids.
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.id.cmp(&other.0.id))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl VectorIndex for BruteForceIndex {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn add(&mut self, vector: &[f32]) -> usize {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        self.data.extend_from_slice(vector);
        self.len() - 1
    }

    fn search(&self, query: &[f32], limit: usize, min_score: f32) -> Vec<Scored> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        if limit == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(limit + 1);
        for (id, row) in self.data.chunks_exact(self.dim).enumerate() {
            let score = dot(query, row);
            if score < min_score {
                continue;
            }
            if heap.len() < limit {
                heap.push(Entry(Scored { id, score }));
            } else if heap.peek().is_some_and(|worst| score > worst.0.score) {
                heap.pop();
                heap.push(Entry(Scored { id, score }));
            }
        }
        let mut out: Vec<Scored> = heap.into_iter().map(|e| e.0).collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_reject_zero_width() {
        assert!(matches!(BruteForceIndex::new(0), Err(Error::Config(_))));
        for capacity in [0, 1, 10] {
            assert!(matches!(
                BruteForceIndex::with_capacity(0, capacity),
                Err(Error::Config(_))
            ));
        }
    }

    #[test]
    fn positive_width_indexes_start_empty_and_accept_vectors() {
        for mut index in [
            BruteForceIndex::new(1).unwrap(),
            BruteForceIndex::with_capacity(1, 10).unwrap(),
        ] {
            assert_eq!(index.dimension(), 1);
            assert!(index.is_empty());
            assert_eq!(index.add(&[1.0]), 0);
            assert_eq!(index.len(), 1);
            assert_eq!(index.search(&[1.0], 1, 0.0), [Scored { id: 0, score: 1.0 }]);
        }
    }

    #[test]
    fn returns_best_first_with_threshold() {
        let mut idx = BruteForceIndex::new(2).unwrap();
        idx.add(&[1.0, 0.0]);
        idx.add(&[0.0, 1.0]);
        idx.add(&[0.6, 0.8]);
        let hits = idx.search(&[1.0, 0.0], 2, -1.0);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), [0, 2]);
        let hits = idx.search(&[1.0, 0.0], 10, 0.5);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn matches_full_sort() {
        let dim = 8;
        let mut idx = BruteForceIndex::new(dim).unwrap();
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / u32::MAX as f32) - 0.25
        };
        let vecs: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..dim).map(|_| rnd()).collect())
            .collect();
        for v in &vecs {
            idx.add(v);
        }
        let q: Vec<f32> = (0..dim).map(|_| rnd()).collect();
        let mut all: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(&q, v)))
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<usize> = idx.search(&q, 10, f32::MIN).iter().map(|s| s.id).collect();
        assert_eq!(top, all[..10].iter().map(|x| x.0).collect::<Vec<_>>());
    }
}
