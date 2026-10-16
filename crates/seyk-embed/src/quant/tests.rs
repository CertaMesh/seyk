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
