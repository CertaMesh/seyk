use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() -> Result<()> {
    // Force a real ORT reference even in the baseline binary.
    let command = std::env::args().nth(1);
    if !matches!(command.as_deref(), Some("render" | "render-only")) {
        let _session = ort::session::Session::builder()?;
    }
    #[cfg(feature = "native-ocr")]
    {
        use geo_clipper::Clipper;
        let poly = geo_types::Polygon::new(
            geo_types::LineString::from(vec![(0., 0.), (4., 0.), (4., 4.), (0., 4.), (0., 0.)]),
            vec![],
        );
        assert_eq!(poly.intersection(&poly, 100.).0.len(), 1);
    }
    match command.as_deref() {
        Some("load-ocr") => {
            for path in std::env::args().skip(2) {
                let now = Instant::now();
                let session = ort::session::Session::builder()?.commit_from_file(&path)?;
                println!(
                    "{}",
                    json!({"asset": path, "load_ms":now.elapsed().as_secs_f64()*1000., "inputs":format!("{:?}",session.inputs()), "outputs":format!("{:?}",session.outputs())})
                );
            }
        }
        #[cfg(feature = "render")]
        Some("render-only") => {
            let mut times = Vec::new();
            for i in 1..=20 {
                let bytes = fs::read(format!("generated/page-{i:02}.pdf"))?;
                let now = Instant::now();
                let (_, _, pixels) = render(bytes, 2048.)?;
                times.push(now.elapsed().as_secs_f64() * 1000.);
                println!(
                    "{}",
                    json!({"page":i,"rgba_sha256":digest(&pixels),"render_ms":times.last()})
                );
            }
            println!("{}", json!({"render":stats(&mut times)}));
        }
        #[cfg(feature = "render")]
        Some("render") => render_corpus()?,
        #[cfg(feature = "embed")]
        Some("embed") => embed_corpus()?,
        _ => println!("{}", json!({"smoke":"PASS", "ort":"2.0.0-rc.13"})),
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn stats(times: &mut [f64]) -> serde_json::Value {
    times.sort_by(f64::total_cmp);
    let n = times.len();
    json!({"n":n,"p50_ms":times[(n as f64 * 0.50).ceil() as usize - 1],"p95_ms":times[(n as f64 * 0.95).ceil() as usize - 1]})
}
fn pdf(objects: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![0];
    for (i, obj) in objects.iter().enumerate() {
        offsets.push(out.len());
        write!(out, "{} 0 obj\n", i + 1).unwrap();
        out.extend(obj);
        out.extend(b"\nendobj\n");
    }
    let start = out.len();
    write!(out, "xref\n0 {}\n0000000000 65535 f \n", offsets.len()).unwrap();
    for offset in offsets.iter().skip(1) {
        writeln!(out, "{offset:010} 00000 n ").unwrap();
    }
    write!(
        out,
        "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{start}\n%%EOF\n",
        offsets.len()
    )
    .unwrap();
    out
}
fn stream(dictionary: &str, content: &[u8]) -> Vec<u8> {
    let mut out = format!("<< {dictionary} /Length {} >>\nstream\n", content.len()).into_bytes();
    out.extend(content);
    out.extend(b"\nendstream");
    out
}
fn page_pdf(content: &[u8], resource: Vec<u8>, page_extra: &str, image: bool) -> Vec<u8> {
    let resources = if image {
        "/XObject << /Im 5 0 R >>"
    } else {
        "/Font << /F1 5 0 R >>"
    };
    pdf(vec![b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(), b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] {page_extra} /Resources << {resources} >> /Contents 4 0 R >>").into_bytes(),stream("",content),resource])
}
#[cfg(feature = "render")]
fn render(bytes: Vec<u8>, long_edge: f32) -> Result<(u32, u32, Vec<u8>)> {
    use hayro::{RenderSettings, hayro_interpret::InterpreterSettings, hayro_syntax::Pdf};
    let document = Pdf::new(std::sync::Arc::new(bytes)).map_err(|e| format!("parse: {e:?}"))?;
    let page = &document.pages()[0];
    let (w, h) = page.render_dimensions();
    let scale = long_edge / w.max(h);
    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        ..Default::default()
    };
    let interpreter = InterpreterSettings {
        warning_sink: std::sync::Arc::new(|warning| panic!("renderer warning: {warning:?}")),
        ..Default::default()
    };
    let pixmap = hayro::render(page, &interpreter, &settings);
    Ok((
        pixmap.width().into(),
        pixmap.height().into(),
        pixmap.data_as_u8_slice().to_vec(),
    ))
}
#[cfg(feature = "render")]
fn render_corpus() -> Result<()> {
    let out = Path::new("generated");
    fs::create_dir_all(out)?;
    let fixture = b"1 0 0 rg 60 100 80 120 re f\n";
    let font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec();
    for (name, extra, dimensions) in [
        ("normal", "", (612, 792)),
        ("rotated", "/Rotate 90", (792, 612)),
        ("cropped", "/CropBox [40 60 400 600]", (360, 540)),
    ] {
        let bytes = page_pdf(fixture, font.clone(), extra, false);
        let edge = dimensions.0.max(dimensions.1) as f32;
        let (w, h, pixels) = render(bytes.clone(), edge)?;
        assert_eq!((w, h), dimensions);
        assert_eq!(pixels, render(bytes, edge)?.2);
        assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);
        assert!(pixels.chunks_exact(4).all(|px| px[3] == 255));
        assert!(pixels.chunks_exact(4).any(|px| px == [255, 0, 0, 255]));
        let red: Vec<_> = pixels
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, p)| *p == [255, 0, 0, 255])
            .map(|(i, _)| (i as u32 % w, i as u32 / w))
            .collect();
        let bounds = (
            red.iter().map(|p| p.0).min().unwrap(),
            red.iter().map(|p| p.1).min().unwrap(),
            red.iter().map(|p| p.0).max().unwrap(),
            red.iter().map(|p| p.1).max().unwrap(),
        );
        let expected = match name {
            "normal" => (60, 572, 139, 691),
            "rotated" => (100, 60, 219, 139),
            "cropped" => (20, 380, 99, 499),
            _ => unreachable!(),
        };
        assert_eq!(bounds, expected, "PDF crop/rotation placement");

        image::save_buffer(
            out.join(format!("{name}.png")),
            &pixels,
            w,
            h,
            image::ColorType::Rgba8,
        )?;
        println!(
            "{}",
            json!({"fixture":name,"width":w,"height":h,"rgba_sha256":digest(&pixels)})
        );
    }
    let corpus: serde_json::Value = serde_json::from_slice(include_bytes!("../corpus.json"))?;
    let mut times = Vec::new();
    for (i, lines) in corpus["pages"].as_array().unwrap().iter().enumerate() {
        let mut content = String::from("BT /F1 20 Tf 50 720 Td 30 TL\n");
        for line in lines.as_array().unwrap() {
            let text = line
                .as_str()
                .unwrap()
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            content.push_str(&format!("({text}) Tj T* /F1 13 Tf\n"));
        }
        content.push_str("ET\n");
        // First create a synthetic scan; the retrieval PDF contains only an image XObject.
        let source = page_pdf(content.as_bytes(), font.clone(), "", false);
        let (w, h, rgba) = render(source, 2200.)?;
        let rgb: Vec<u8> = rgba
            .chunks_exact(4)
            .flat_map(|p| p[..3].iter().copied())
            .collect();
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&rgb)?;
        let image = stream(
            &format!(
                "/Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode"
            ),
            &z.finish()?,
        );
        let scan = page_pdf(b"q 612 0 0 792 0 0 cm /Im Do Q", image, "", true);
        fs::write(out.join(format!("page-{:02}.pdf", i + 1)), &scan)?;
        let now = Instant::now();
        let (rw, rh, pixels) = render(scan.clone(), 2048.)?;
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(rw, rh, pixels.clone()).unwrap(),
        )
        .write_to(&mut png, image::ImageFormat::Png)?;
        times.push(now.elapsed().as_secs_f64() * 1000.);
        assert_eq!(pixels, render(scan, 2048.)?.2);
        let bytes = png.into_inner();
        fs::write(out.join(format!("page-{:02}.png", i + 1)), &bytes)?;
        println!(
            "{}",
            json!({"page":i+1,"png_sha256":digest(&bytes),"bytes":bytes.len(),"width":rw,"height":rh,"render_encode_ms":times.last()})
        );
    }
    println!("{}", json!({"render_encode":stats(&mut times)}));
    Ok(())
}
#[cfg(feature = "embed")]
fn embed_corpus() -> Result<()> {
    use seyk_embed::{Embedder, EmbeddingTask, ResourceInput, cosine_similarity};
    let corpus: serde_json::Value = serde_json::from_slice(include_bytes!("../corpus.json"))?;
    let now = Instant::now();
    // The model, precision, dimension and thread count all keep Seyk defaults.
    let model = seyk_embeddinggemma::EmbeddingGemma2::builder()
        .vision(true)
        .build()?;
    println!(
        "{}",
        json!({"load_ms":now.elapsed().as_secs_f64()*1000.,"dimension":model.config().dimension.get(),"precision":model.config().precision.name(),"threads":model.config().threads,"revision":seyk_embeddinggemma::ONNX_REVISION})
    );
    let mut vectors = Vec::new();
    let mut times = Vec::new();
    for i in 1..=20 {
        let bytes = fs::read(format!("generated/page-{i:02}.png"))?;
        let now = Instant::now();
        vectors.push(model.embed(
            ResourceInput::Image { bytes: &bytes },
            EmbeddingTask::SearchDocument,
        )?);
        times.push(now.elapsed().as_secs_f64() * 1000.);
        println!("{}", json!({"page":i,"embed_ms":times.last()}));
    }
    let mut recalls = 0;
    let mut mrr = 0.;
    let mut query_times = Vec::new();
    for q in corpus["queries"].as_array().unwrap() {
        let text = q[0].as_str().unwrap();
        let target = q[1].as_u64().unwrap() as usize;
        let now = Instant::now();
        let query = model.embed_text(text, EmbeddingTask::SearchQuery)?;
        query_times.push(now.elapsed().as_secs_f64() * 1000.);
        let mut ranked: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i + 1, cosine_similarity(&query, v).unwrap()))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rank = ranked.iter().position(|(i, _)| *i == target).unwrap() + 1;
        if rank <= 5 {
            recalls += 1;
        }
        if rank <= 10 {
            mrr += 1. / rank as f64;
        }
        println!(
            "{}",
            json!({"query":text,"target":target,"rank":rank,"top5":&ranked[..5]})
        );
    }
    let n = corpus["queries"].as_array().unwrap().len();
    println!(
        "{}",
        json!({"queries":n,"recall_at_5":recalls as f64/n as f64,"mrr_at_10":mrr/n as f64,"embed":stats(&mut times),"query":stats(&mut query_times)})
    );
    Ok(())
}
