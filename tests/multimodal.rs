#![cfg(all(feature = "onnx", feature = "image"))]

use std::path::{Path, PathBuf};

use gemma_search::Dimension;
use gemma_search::SearchEngine;
use gemma_search::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision, vision};
use gemma_search::embedder::{Embedder, ResourceInput};
use image::{ImageBuffer, Rgb};
use serde_json::Value;

fn rgb(r: u8, g: u8, b: u8) -> Rgb<u8> {
    Rgb([r, g, b])
}

fn write_red_circle(path: &Path) {
    let mut image = ImageBuffer::from_pixel(192, 192, rgb(225, 240, 250));
    for y in 40..153 {
        for x in 40..153 {
            let dx = x as i32 - 96;
            let dy = y as i32 - 96;
            if dx * dx + dy * dy <= 56 * 56 {
                image.put_pixel(x, y, rgb(220, 35, 35));
            }
        }
    }
    image.save(path).unwrap();
}

fn write_bar_chart(path: &Path) {
    let mut image = ImageBuffer::from_pixel(480, 192, rgb(255, 255, 255));
    for x in 48..445 {
        image.put_pixel(x, 152, rgb(35, 35, 35));
    }
    for y in 32..153 {
        image.put_pixel(48, y, rgb(35, 35, 35));
    }
    let heights = [48, 82, 64, 118, 96, 143];
    let colors = [
        rgb(45, 110, 205),
        rgb(55, 145, 210),
        rgb(70, 175, 145),
        rgb(245, 165, 45),
        rgb(230, 100, 65),
        rgb(170, 75, 170),
    ];
    for (i, (&height, color)) in heights.iter().zip(colors).enumerate() {
        let x0 = 68 + i as u32 * 60;
        for y in (152 - height)..152 {
            for x in x0..x0 + 34 {
                image.put_pixel(x, y, color);
            }
        }
    }
    image.save(path).unwrap();
}

fn write_text_page(path: &Path) {
    let mut image = ImageBuffer::from_pixel(192, 480, rgb(248, 246, 238));
    for x in 18..175 {
        image.put_pixel(x, 18, rgb(80, 80, 75));
        image.put_pixel(x, 462, rgb(80, 80, 75));
    }
    for y in 18..463 {
        image.put_pixel(18, y, rgb(80, 80, 75));
        image.put_pixel(174, y, rgb(80, 80, 75));
    }
    for y in [45, 50, 55, 75] {
        for x in 38..154 {
            image.put_pixel(x, y, rgb(35, 70, 100));
        }
    }
    for row in 0..9 {
        let y = 102 + row * 34;
        for x in 62..157 {
            image.put_pixel(x, y + 7, rgb(105, 105, 105));
            image.put_pixel(x, y + 8, rgb(105, 105, 105));
        }
    }
    image.save(path).unwrap();
}

fn first_path(results: &gemma_search::SearchResults) -> PathBuf {
    results
        .hits
        .first()
        .expect("cross-modal query should return a hit")
        .resource
        .path()
        .to_path_buf()
}

fn json_f32s(value: &Value) -> Vec<f32> {
    if let Some(values) = value.as_array() {
        values.iter().flat_map(json_f32s).collect()
    } else {
        vec![value.as_f64().unwrap() as f32]
    }
}

#[test]
#[ignore = "needs generated Python reference tensors"]
fn image_preprocessing_matches_python_reference() {
    let reference_path = std::env::var("GEMMA_IMAGE_REFERENCE_JSON")
        .expect("set GEMMA_IMAGE_REFERENCE_JSON to the Python reference JSON");
    let model_dir =
        std::env::var("GEMMA_ONNX_DIR").expect("set GEMMA_ONNX_DIR to the ONNX snapshot directory");
    let root: Value = serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&model_dir).join("tokenizer.json")).unwrap();

    for record in root["images"].as_array().unwrap() {
        let path = PathBuf::from(record["path"].as_str().unwrap());
        let bytes = std::fs::read(&path).unwrap();
        let pre = vision::preprocess(&bytes).unwrap();
        let expected_size = record["resized_size"].as_array().unwrap();
        assert_eq!(pre.width, expected_size[0].as_u64().unwrap() as u32);
        assert_eq!(pre.height, expected_size[1].as_u64().unwrap() as u32);
        assert_eq!(
            pre.soft_tokens,
            record["num_soft_tokens"].as_u64().unwrap() as usize
        );

        let expected_pixels = json_f32s(&record["pixel_values"]);
        assert_eq!(pre.pixel_values.len(), expected_pixels.len());
        let max_pixel_diff = pre
            .pixel_values
            .iter()
            .zip(&expected_pixels)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let expected_positions = record["position_ids"].as_array().unwrap();
        let positions_match = pre
            .position_ids
            .chunks_exact(2)
            .zip(expected_positions)
            .all(|(actual, expected)| {
                actual[0] == expected[0].as_i64().unwrap()
                    && actual[1] == expected[1].as_i64().unwrap()
            });
        assert!(
            positions_match,
            "position ids differ for {}",
            path.display()
        );

        let text = format!("<|image>{}<image|>", "<|image|>".repeat(pre.soft_tokens));
        let actual_ids = tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        let expected_ids: Vec<u32> = record["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            actual_ids,
            expected_ids,
            "token ids differ for {}",
            path.display()
        );
        println!(
            "{} resized={}x{} soft_tokens={} max_pixel_diff={:.7}",
            record["name"].as_str().unwrap(),
            pre.width,
            pre.height,
            pre.soft_tokens,
            max_pixel_diff
        );
    }
}

#[test]
#[ignore = "needs generated Python reference tensors and EmbeddingGemma 2 weights"]
fn image_preprocessing_and_embedding_matches_python_reference() {
    let reference_path = std::env::var("GEMMA_IMAGE_REFERENCE_JSON")
        .expect("set GEMMA_IMAGE_REFERENCE_JSON to the Python reference JSON");
    let model_dir =
        std::env::var("GEMMA_ONNX_DIR").expect("set GEMMA_ONNX_DIR to the ONNX snapshot directory");
    let precision: Precision = std::env::var("GEMMA_SEARCH_TEST_PRECISION")
        .unwrap_or_else(|_| "q8".into())
        .parse()
        .unwrap();
    let root: Value = serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let embedder = GemmaEmbedder::load(GemmaConfig {
        source: ModelSource::Directory(model_dir.clone().into()),
        dimension: Dimension::D768,
        precision,
        vision: true,
        threads: Some(1),
        batch_size: 1,
        ..GemmaConfig::default()
    })
    .unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(Path::new(&model_dir).join("tokenizer.json")).unwrap();

    let cosine_floor = if precision == Precision::Fp32 {
        0.999
    } else {
        0.995
    };
    for record in root["images"].as_array().unwrap() {
        let path = PathBuf::from(record["path"].as_str().unwrap());
        let bytes = std::fs::read(&path).unwrap();
        let pre = vision::preprocess(&bytes).unwrap();
        let expected_size = record["resized_size"].as_array().unwrap();
        assert_eq!(pre.width, expected_size[0].as_u64().unwrap() as u32);
        assert_eq!(pre.height, expected_size[1].as_u64().unwrap() as u32);
        assert_eq!(
            pre.soft_tokens,
            record["num_soft_tokens"].as_u64().unwrap() as usize
        );

        let expected_pixels = json_f32s(&record["pixel_values"]);
        let max_pixel_diff = pre
            .pixel_values
            .iter()
            .zip(&expected_pixels)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(pre.pixel_values.len(), expected_pixels.len());
        let expected_positions = record["position_ids"].as_array().unwrap();
        let positions_match = pre
            .position_ids
            .chunks_exact(2)
            .zip(expected_positions)
            .all(|(actual, expected)| {
                actual[0] == expected[0].as_i64().unwrap()
                    && actual[1] == expected[1].as_i64().unwrap()
            });
        assert!(
            positions_match,
            "position ids differ for {}",
            path.display()
        );

        let text = format!("<|image>{}<image|>", "<|image|>".repeat(pre.soft_tokens));
        let actual_ids = tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        let expected_ids: Vec<u32> = record["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            actual_ids,
            expected_ids,
            "token ids differ for {}",
            path.display()
        );

        let actual = embedder
            .embed_resources(&[ResourceInput::Image { bytes: &bytes }])
            .unwrap()
            .pop()
            .unwrap();
        let expected =
            gemma_search::Embedding::from_raw(json_f32s(&record["embedding"]), Dimension::D768)
                .unwrap();
        let cosine = actual.cosine(&expected);
        println!(
            "{} precision={} resized={}x{} soft_tokens={} max_pixel_diff={:.7} cosine={:.7}",
            record["name"].as_str().unwrap(),
            precision.name(),
            pre.width,
            pre.height,
            pre.soft_tokens,
            max_pixel_diff,
            cosine
        );
        assert!(
            cosine >= cosine_floor,
            "{} cosine {cosine:.7} below {cosine_floor}",
            path.display()
        );
    }
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn text_queries_retrieve_generated_images_and_mixed_content() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let red = root.join("red-circle.png");
    let chart = root.join("bar-chart.png");
    let page = root.join("text-page.png");
    write_red_circle(&red);
    write_bar_chart(&chart);
    write_text_page(&page);
    std::fs::write(
        root.join("context.md"),
        "# Unrelated context\n\nThis note describes a quiet forest walk and a cup of tea.\n",
    )
    .unwrap();

    let embedder = GemmaEmbedder::load(GemmaConfig {
        threads: Some(1),
        batch_size: 1,
        ..GemmaConfig::default()
    })
    .unwrap();
    let engine = SearchEngine::builder().embedder(embedder).build().unwrap();

    let red = red.canonicalize().unwrap();
    let chart = chart.canonicalize().unwrap();
    let page = page.canonicalize().unwrap();
    let paths = [root.to_path_buf()];

    let result = engine
        .search("a red circle")
        .paths(paths.clone())
        .limit(4)
        .execute()
        .unwrap();
    assert_eq!(first_path(&result), red);

    let result = engine
        .search("a colored bar chart with rising columns")
        .paths(paths.clone())
        .limit(4)
        .execute()
        .unwrap();
    assert_eq!(first_path(&result), chart);

    let result = engine
        .search("a page of written notes")
        .paths(paths.clone())
        .limit(4)
        .execute()
        .unwrap();
    assert_eq!(first_path(&result), page);

    let mixed = engine
        .search("mountain landscape")
        .paths(paths)
        .limit(4)
        .execute()
        .unwrap();
    assert_eq!(mixed.stats.sync.resources_discovered, 4);
    assert!(mixed.stats.sync.skipped.is_empty());
    assert_eq!(mixed.stats.segments_searched, 4);
}
