#![cfg(all(feature = "onnx", feature = "image"))]

use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb};
use serde::Deserialize;
use seyk::Dimension;
use seyk::SearchEngine;
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision, vision};
use seyk::embedder::{Embedder, ResourceInput};

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

fn first_path(results: &seyk::SearchResults) -> PathBuf {
    results
        .hits
        .first()
        .expect("cross-modal query should return a hit")
        .resource
        .to_file_path()
        .expect("file resource")
        .to_path_buf()
}

/// Reference data from scripts/reference_image_embeddings.py.
#[derive(Deserialize)]
struct ImageReference {
    images: Vec<ImageRecord>,
}

#[derive(Deserialize)]
struct ImageRecord {
    name: String,
    path: PathBuf,
    /// (width, height)
    resized_size: [u32; 2],
    pixel_values_shape: [usize; 2],
    pixel_values: Vec<Vec<f32>>,
    position_ids_shape: [usize; 2],
    position_ids: Vec<[i64; 2]>,
    num_soft_tokens: usize,
    input_ids: Vec<u32>,
    embedding: Vec<f32>,
}

impl ImageRecord {
    /// Pixel patches flattened row-major, after checking them against the declared shape.
    fn pixels(&self) -> Result<Vec<f32>, String> {
        let [rows, width] = self.pixel_values_shape;
        if self.pixel_values.len() != rows || self.pixel_values.iter().any(|r| r.len() != width) {
            return Err(format!(
                "{}: pixel values do not match [{rows}, {width}]",
                self.name
            ));
        }
        Ok(self.pixel_values.concat())
    }

    /// Patch positions flattened as (x, y) pairs, after checking the declared shape.
    fn positions(&self) -> Result<Vec<i64>, String> {
        let rows = self.position_ids.len();
        if self.position_ids_shape != [rows, 2] {
            return Err(format!(
                "{}: {rows} position rows, declared {:?}",
                self.name, self.position_ids_shape
            ));
        }
        Ok(self.position_ids.concat())
    }
}

fn load_reference() -> ImageReference {
    let path = std::env::var("SEYK_IMAGE_REFERENCE_JSON")
        .expect("set SEYK_IMAGE_REFERENCE_JSON to the Python reference JSON");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn load_tokenizer(model_dir: &str) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(Path::new(model_dir).join("tokenizer.json")).unwrap()
}

/// Checks Rust preprocessing and tokenisation against one reference record.
/// Returns the image bytes, the preprocessed image and the largest pixel
/// difference, which is reported rather than asserted.
fn check_preprocessing(
    record: &ImageRecord,
    tokenizer: &tokenizers::Tokenizer,
) -> (Vec<u8>, vision::Preprocessed, f32) {
    let path = record.path.display();
    let bytes = std::fs::read(&record.path).unwrap();
    let pre = vision::preprocess(&bytes).unwrap();
    assert_eq!([pre.width, pre.height], record.resized_size, "{path}");
    assert_eq!(pre.soft_tokens, record.num_soft_tokens, "{path}");

    let expected_pixels = record.pixels().unwrap();
    assert_eq!(pre.pixel_values.len(), expected_pixels.len(), "{path}");
    let max_pixel_diff = pre
        .pixel_values
        .iter()
        .zip(&expected_pixels)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        pre.position_ids,
        record.positions().unwrap(),
        "position ids differ for {path}"
    );

    let text = format!("<|image>{}<image|>", "<|image|>".repeat(pre.soft_tokens));
    let actual_ids = tokenizer.encode(text, true).unwrap().get_ids().to_vec();
    assert_eq!(actual_ids, record.input_ids, "token ids differ for {path}");
    (bytes, pre, max_pixel_diff)
}

#[test]
fn image_reference_records_check_declared_shapes() {
    let json = r#"{"images": [{
        "name": "tiny", "path": "tiny.png", "original_size": [2, 1],
        "resized_size": [2, 1],
        "pixel_values_shape": [2, 3], "pixel_values": [[0, 1, 2], [3, 4, 5]],
        "position_ids_shape": [2, 2], "position_ids": [[0, 0], [-1, -1]],
        "num_soft_tokens": 1, "input_ids": [1, 2], "embedding": [0.5]
    }]}"#;
    let mut record = serde_json::from_str::<ImageReference>(json)
        .unwrap()
        .images
        .remove(0);
    assert_eq!(record.pixels().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(record.positions().unwrap(), [0, 0, -1, -1]);

    record.position_ids.pop();
    assert!(record.positions().is_err());
    record.pixel_values[1].pop();
    assert!(record.pixels().is_err());
    let triple = json.replace("[-1, -1]", "[-1, -1, -1]");
    assert!(serde_json::from_str::<ImageReference>(&triple).is_err());
}

#[test]
#[ignore = "needs generated Python reference tensors"]
fn image_preprocessing_matches_python_reference() {
    let model_dir =
        std::env::var("SEYK_ONNX_DIR").expect("set SEYK_ONNX_DIR to the ONNX snapshot directory");
    let reference = load_reference();
    let tokenizer = load_tokenizer(&model_dir);

    for record in &reference.images {
        let (_, pre, max_pixel_diff) = check_preprocessing(record, &tokenizer);
        println!(
            "{} resized={}x{} soft_tokens={} max_pixel_diff={:.7}",
            record.name, pre.width, pre.height, pre.soft_tokens, max_pixel_diff
        );
    }
}

#[test]
#[ignore = "needs generated Python reference tensors and EmbeddingGemma 2 weights"]
fn image_preprocessing_and_embedding_matches_python_reference() {
    let model_dir =
        std::env::var("SEYK_ONNX_DIR").expect("set SEYK_ONNX_DIR to the ONNX snapshot directory");
    let precision: Precision = std::env::var("SEYK_TEST_PRECISION")
        .unwrap_or_else(|_| "q8".into())
        .parse()
        .unwrap();
    let reference = load_reference();
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
    let tokenizer = load_tokenizer(&model_dir);

    let cosine_floor = if precision == Precision::Fp32 {
        0.999
    } else {
        0.995
    };
    for record in &reference.images {
        let (bytes, pre, max_pixel_diff) = check_preprocessing(record, &tokenizer);
        let actual = embedder
            .embed_resources(&[ResourceInput::Image { bytes: &bytes }])
            .unwrap()
            .pop()
            .unwrap();
        let expected =
            seyk::Embedding::from_raw(record.embedding.clone(), Dimension::D768).unwrap();
        let cosine = actual.cosine(&expected);
        println!(
            "{} precision={} resized={}x{} soft_tokens={} max_pixel_diff={:.7} cosine={:.7}",
            record.name,
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
            record.path.display()
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
