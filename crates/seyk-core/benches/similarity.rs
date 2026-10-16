//! Similarity search and cache load benchmarks on synthetic vectors (no model needed).
//!
//! `cargo bench --bench similarity`. Filter with e.g. `-- brute_force/768`.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use seyk_core::cache::{CacheProfile, CachedResource, CachedSegment, ResourceWrite, SegmentWrite};
use seyk_core::index::{BruteForceIndex, VectorIndex};
use seyk_core::{
    Dimension, EmbeddingCache, Modality, ModelIdentity, ResourceId, ResourceLocator,
    ResourceMetadata, ResourceVersion, SqliteCache,
};

const SIZES: [usize; 4] = [100, 1_000, 10_000, 100_000];
const LIMIT: usize = 10;
const SEGMENTS_PER_FILE: usize = 10;

/// Deterministic pseudo-random unit vectors, so runs are comparable.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }
}

fn brute_force(c: &mut Criterion) {
    for dim in [256, 768] {
        let mut group = c.benchmark_group(format!("brute_force/{dim}"));
        group
            .sample_size(30)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(3));
        let mut rng = Rng(dim as u64);
        let query = rng.unit_vector(dim);
        for n in SIZES {
            let mut index = BruteForceIndex::with_capacity(dim, n).unwrap();
            for _ in 0..n {
                index.add(&rng.unit_vector(dim));
            }
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("top10", n), &index, |b, index| {
                b.iter(|| index.search(black_box(&query), LIMIT, f32::MIN))
            });
        }
        group.finish();
    }
}

/// Reference kernel with eight independent accumulators, matching the library's
/// lane-based reduction so the benchmark can compare equivalent kernels.
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn dot_kernel(c: &mut Criterion) {
    let n = 10_000;
    for dim in [256, 768] {
        let mut group = c.benchmark_group(format!("dot_kernel/{dim}"));
        group
            .sample_size(30)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(n as u64));
        let mut rng = Rng(7);
        let query = rng.unit_vector(dim);
        let matrix: Vec<f32> = (0..n).flat_map(|_| rng.unit_vector(dim)).collect();
        group.bench_function(BenchmarkId::new("sequential_sum", n), |b| {
            b.iter(|| {
                matrix
                    .chunks_exact(dim)
                    .map(|row| row.iter().zip(&query).map(|(x, y)| x * y).sum::<f32>())
                    .fold(f32::MIN, f32::max)
            })
        });
        group.bench_function(BenchmarkId::new("eight_lanes", n), |b| {
            b.iter(|| {
                matrix
                    .chunks_exact(dim)
                    .map(|row| dot_lanes(row, black_box(&query)))
                    .fold(f32::MIN, f32::max)
            })
        });
        group.finish();
    }
}

fn profile(dim: Dimension) -> CacheProfile {
    CacheProfile {
        model: ModelIdentity {
            model_id: "bench/synthetic".into(),
            revision: "0".into(),
            variant: "synthetic".into(),
            dimension: dim,
            normalization: "l2".into(),
            query_prompt: String::new(),
            document_prompt: String::new(),
            max_tokens: 0,
        },
        segmenter_version: "bench".into(),
    }
}

/// A cache holding `n` segments of `dim` floats spread over `n / SEGMENTS_PER_FILE` files.
fn filled_cache(
    dir: &std::path::Path,
    n: usize,
    dim: Dimension,
) -> (SqliteCache, String, HashSet<ResourceId>) {
    let mut cache = SqliteCache::open(dir.join(format!("cache-{n}.sqlite3"))).unwrap();
    let profile = profile(dim);
    let pid = profile.id();
    cache.register_profile(&pid, &profile).unwrap();
    let mut rng = Rng(n as u64);
    let mut paths = HashSet::new();
    for f in 0..n.div_ceil(SEGMENTS_PER_FILE) {
        let path = ResourceId::from_path(format!("/notes/chapter-{f:06}.md")).unwrap();
        let segments: Vec<CachedSegment> = (0..SEGMENTS_PER_FILE)
            .map(|s| CachedSegment {
                locator: ResourceLocator::Text {
                    byte_start: s as u64 * 1000,
                    byte_end: (s as u64 + 1) * 1000,
                    line_start: Some(s as u64 * 20 + 1),
                    line_end: Some((s as u64 + 1) * 20),
                },
                heading: Some(format!("Section {s}")),
                input_hash: format!("{f:06}-{s:02}"),
            })
            .collect();
        let vectors: Vec<(String, Vec<f32>)> = segments
            .iter()
            .map(|s| (s.input_hash.clone(), rng.unit_vector(dim.get())))
            .collect();
        let resource = CachedResource {
            descriptor_hash: None,
            version: ResourceVersion::known("fs:10000:0"),
            metadata: ResourceMetadata {
                media_type: None,
                modality: Modality::Text,
                size_bytes: 10_000,
                modified_ns: Some(0),
                content_hash: format!("{f:06}"),
            },
            segments,
        };
        cache
            .put_resource(
                &pid,
                &path,
                &ResourceWrite {
                    metadata: &resource.metadata,
                    version: &resource.version,
                    descriptor_hash: resource.descriptor_hash.as_deref(),
                    segments: resource
                        .segments
                        .iter()
                        .map(|segment| SegmentWrite {
                            segment,
                            lexical_body: "",
                        })
                        .collect(),
                },
            )
            .unwrap();
        cache.put_embeddings(&pid, &vectors).unwrap();
        paths.insert(path);
    }
    (cache, pid, paths)
}

fn cache_load(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let dim = Dimension::D256;
    let mut group = c.benchmark_group(format!("sqlite_vectors_for/{dim}"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(5));
    for n in [1_000, 10_000, 100_000] {
        let (cache, pid, paths) = filled_cache(dir.path(), n, dim);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("load", n), |b| {
            b.iter(|| {
                let v = cache.vectors_for(&pid, black_box(&paths)).unwrap();
                assert_eq!(v.len(), n);
                v
            })
        });
    }
    group.finish();
}

fn photo_anchor_scan(c: &mut Criterion) {
    use seyk_embed::{
        Embedding, dot_product,
        quant::{PhotoAnchorQ4, QueryLut},
    };
    for dim in [256, 768] {
        let codec = PhotoAnchorQ4::new(dim).unwrap();
        let mut rng = Rng(4517);
        let dimension = Dimension::try_from(dim).unwrap();
        let query = Embedding::from_raw(rng.unit_vector(dim), dimension).unwrap();
        let lut = QueryLut::new(&query).unwrap();
        let rows: Vec<_> = (0..1_000)
            .map(|_| Embedding::from_raw(rng.unit_vector(dim), dimension).unwrap())
            .collect();
        let codes: Vec<u8> = rows
            .iter()
            .flat_map(|row| codec.encode(row).unwrap())
            .collect();
        let floats: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.as_slice().iter().copied())
            .collect();
        let mut group = c.benchmark_group(format!("photo_anchor_scan/{dim}"));
        group
            .sample_size(30)
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(1_000));
        // Both paths scan all 1k rows and consume every score. Setup, allocation,
        // encoding and LUT construction are excluded from the timed region.
        group.bench_function("f32", |b| {
            b.iter(|| {
                black_box(&floats)
                    .chunks_exact(dim)
                    .map(|row| dot_product(row, black_box(query.as_slice())).unwrap())
                    .fold(f32::MIN, f32::max)
            })
        });
        group.bench_function("q4_lut", |b| {
            b.iter(|| {
                black_box(&codes)
                    .chunks_exact(dim / 2)
                    .map(|row| black_box(&lut).score(row).unwrap())
                    .fold(f32::MIN, f32::max)
            })
        });
        group.finish();
    }
}

criterion_group!(
    benches,
    brute_force,
    dot_kernel,
    cache_load,
    photo_anchor_scan
);
criterion_main!(benches);
