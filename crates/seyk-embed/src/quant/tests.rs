use super::*;
use crate::Dimension;

fn vector(dim: usize, seed: usize) -> Embedding {
    let mut v = vec![0.0; dim];
    if seed == 3 {
        v[0] = 1.0;
    } else {
        for i in 0..256 {
            v[(i * 5 + seed * 11) % dim] = if (i * 17 + seed * 13) % 7 < 3 {
                1.0 / 16.0
            } else {
                -1.0 / 16.0
            };
        }
    }
    Embedding::from_raw(v, Dimension::try_from(dim).unwrap()).unwrap()
}

#[test]
fn python_codes_and_scores_match() {
    #[derive(serde::Deserialize)]
    struct Golden {
        dim: usize,
        seed: usize,
        codes: Vec<u8>,
        score: f32,
    }
    let goldens: Vec<Golden> =
        serde_json::from_str(include_str!("../../assets/photo-anchor/golden.json")).unwrap();
    for golden in goldens {
        let codec = PhotoAnchorQ4::new(golden.dim).unwrap();
        let codes = codec.encode(&vector(golden.dim, golden.seed)).unwrap();
        assert_eq!(
            codes, golden.codes,
            "dim={}, seed={}",
            golden.dim, golden.seed
        );
        let query = vector(golden.dim, 2);
        let actual = QueryLut::new(&query).unwrap().score(&codes).unwrap();
        assert!((actual - golden.score).abs() < 1e-5);
        let naive: f32 = (0..golden.dim)
            .map(|i| {
                let code = if i % 2 == 0 {
                    codes[i / 2] >> 4
                } else {
                    codes[i / 2] & 15
                };
                (codec.anchor[i] + codec.offset[i] + f32::from(code) * codec.scale[i])
                    * query.as_slice()[i]
            })
            .sum();
        assert!((actual - naive).abs() < 1e-5);
    }
}

#[test]
fn ties_to_even_clipping_and_high_nibble_first() {
    // Dyadic inputs make the half steps exact, exercising the encoder itself.
    let mut raw = vec![0.0; 256];
    raw[0] = 1.0;
    let input = Embedding::from_raw(raw, Dimension::D256).unwrap();
    for (step, expected) in [
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),
        (3.5, 4),
        (14.5, 14),
        (15.5, 15),
        (-0.5, 0),
    ] {
        let mut codec = PhotoAnchorQ4 {
            anchor: vec![0.0; 256],
            offset: vec![0.0; 256],
            scale: vec![1.0; 256],
        };
        codec.anchor[0] = 1.0 - step;
        codec.anchor[1] = -3.0;
        assert_eq!(codec.encode(&input).unwrap()[0], expected << 4 | 3);
    }
}

#[test]
fn every_packed_value_scores_like_naive_decode() {
    for dim in [256, 768] {
        let codec = PhotoAnchorQ4::new(dim).unwrap();
        let query = vector(dim, 0);
        let lut = QueryLut::new(&query).unwrap();
        for byte in 0..=255u8 {
            let codes = vec![byte; dim / 2];
            let naive: f32 = (0..dim)
                .map(|i| {
                    let code = if i % 2 == 0 { byte >> 4 } else { byte & 15 };
                    (codec.anchor[i] + codec.offset[i] + f32::from(code) * codec.scale[i])
                        * query.as_slice()[i]
                })
                .sum();
            assert!((lut.score(&codes).unwrap() - naive).abs() < 1e-5);
        }
    }
}

#[test]
fn rejects_invalid_inputs_and_corrupted_assets() {
    for dim in [0, 128, 255, 512, 769] {
        assert!(matches!(PhotoAnchorQ4::new(dim), Err(Error::UnsupportedDimension(d)) if d == dim));
    }
    let codec = PhotoAnchorQ4::new(256).unwrap();
    assert!(matches!(
        codec.encode(&vector(768, 0)),
        Err(Error::Length { .. })
    ));
    for value in [0.0, 2.0] {
        let input =
            serde_json::from_value::<Embedding>(serde_json::json!(vec![value; 256])).unwrap();
        assert_eq!(codec.encode(&input), Err(Error::NonUnitEmbedding));
        assert!(matches!(
            QueryLut::new(&input),
            Err(Error::NonUnitEmbedding)
        ));
    }
    let lut = QueryLut::new(&vector(256, 0)).unwrap();
    for len in [0, 127, 129] {
        assert!(matches!(
            lut.score(&vec![0; len]),
            Err(Error::Length { .. })
        ));
    }
    let anchor = include_bytes!("../../assets/photo-anchor/photo_anchor_256.f32.bin");
    let table = include_bytes!("../../assets/photo-anchor/quant4_256.f32.bin");
    for corrupt in 0..3 {
        let mut assets = [MANIFEST.to_vec(), anchor.to_vec(), table.to_vec()];
        assets[corrupt][0] ^= 1;
        assert!(matches!(
            PhotoAnchorQ4::load(
                256,
                &assets[0],
                &assets[1],
                &assets[2],
                &sha256::hex(anchor),
                &sha256::hex(table)
            ),
            Err(Error::AssetIntegrity(_))
        ));
    }
}

// These references deliberately parse the pinned files without the production loader.
fn independent_scores(query: &[f32], codes: &[u8]) -> (f32, f64) {
    let dim = query.len();
    let (anchor, table): (&[u8], &[u8]) = match dim {
        256 => (
            include_bytes!("../../assets/photo-anchor/photo_anchor_256.f32.bin"),
            include_bytes!("../../assets/photo-anchor/quant4_256.f32.bin"),
        ),
        768 => (
            include_bytes!("../../assets/photo-anchor/photo_anchor_768.f32.bin"),
            include_bytes!("../../assets/photo-anchor/quant4_768.f32.bin"),
        ),
        _ => panic!("unsupported test dimension"),
    };
    let read =
        |bytes: &[u8], i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    let mut sequential = 0.0f32;
    let mut double = 0.0f64;
    for (i, &q) in query.iter().enumerate() {
        let code = if i % 2 == 0 {
            codes[i / 2] >> 4
        } else {
            codes[i / 2] & 15
        };
        let a = read(anchor, i);
        let o = read(table, i);
        let s = read(table, dim + i);
        sequential += (a + o + f32::from(code) * s) * q;
        double += (f64::from(a) + f64::from(o) + f64::from(code) * f64::from(s)) * f64::from(q);
    }
    (sequential, double)
}

fn assert_independent_contract(query: &Embedding, codes: &[u8]) {
    let actual = QueryLut::new(query).unwrap().score(codes).unwrap();
    let (sequential, double) = independent_scores(query.as_slice(), codes);
    let tolerance: f64 = if query.dim() == 256 { 5e-5 } else { 1.5e-4 };
    assert!(
        (f64::from(actual) - f64::from(sequential)).abs() <= tolerance,
        "d{} LUT {actual} sequential {sequential}",
        query.dim()
    );
    assert!(
        (f64::from(actual) - double).abs() <= tolerance,
        "d{} LUT {actual} f64 {double}",
        query.dim()
    );
}

#[test]
fn absolute_accuracy_against_independent_f32_and_f64_references() {
    let mut state = 4517u64;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    for dim in [256, 768] {
        let queries: Vec<_> = (0..8)
            .map(|seed| {
                if seed < 4 {
                    vector(dim, seed)
                } else {
                    Embedding::from_raw(
                        (0..dim).map(|_| random()).collect(),
                        Dimension::try_from(dim).unwrap(),
                    )
                    .unwrap()
                }
            })
            .collect();
        for query in &queries {
            for byte in 0..=255u8 {
                assert_independent_contract(query, &vec![byte; dim / 2]);
            }
            for _ in 0..64 {
                let row: Vec<_> = (0..dim / 2)
                    .map(|_| (random() * 127.0 + 128.0) as u8)
                    .collect();
                assert_independent_contract(query, &row);
            }
            for seed in 0..4 {
                let codes = PhotoAnchorQ4::new(dim)
                    .unwrap()
                    .encode(&vector(dim, seed))
                    .unwrap();
                assert_independent_contract(query, &codes);
            }
        }
    }
}

#[test]
fn preserved_rounding_witness_satisfies_the_amended_absolute_contract() {
    let witness: serde_json::Value = serde_json::from_str(include_str!(
        "../../assets/photo-anchor/rounding-witness.json"
    ))
    .unwrap();
    let values: Vec<f32> = witness["q_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()))
        .collect();
    let query: Embedding = serde_json::from_value(serde_json::to_value(values).unwrap()).unwrap();
    assert!(query.normalized());
    let hex = witness["row_hex"].as_str().unwrap();
    let codes: Vec<_> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let actual = QueryLut::new(&query).unwrap().score(&codes).unwrap();
    let (sequential, _) = independent_scores(query.as_slice(), &codes);
    assert_eq!(actual, witness["lut"].as_f64().unwrap() as f32);
    assert_eq!(sequential, witness["decode"].as_f64().unwrap() as f32);
    assert_independent_contract(&query, &codes);
}
