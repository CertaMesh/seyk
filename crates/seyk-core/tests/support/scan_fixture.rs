#![allow(dead_code)]
//! Wholly synthetic fixtures. No source text or third-party PDFs are bundled.
use std::io::Write;

fn stream(bytes: &[u8], dictionary: &str) -> Vec<u8> {
    let mut out = format!("<< {dictionary} /Length {} >>\nstream\n", bytes.len()).into_bytes();
    out.extend(bytes);
    out.extend(b"\nendstream");
    out
}

pub fn document(pages: &[(&[u8], &str)], image: Option<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!(
            "<< /Type /Pages /Count {} /Kids [{}] >>",
            pages.len(),
            (0..pages.len())
                .map(|i| format!("{} 0 R", 5 + i * 2))
                .collect::<Vec<_>>()
                .join(" ")
        )
        .into_bytes(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
        image.unwrap_or_else(|| b"null".to_vec()),
    ];
    for (i, (content, extra)) in pages.iter().enumerate() {
        objects.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] {extra} /Resources << /Font << /F1 3 0 R >> /XObject << /Im 4 0 R >> >> /Contents {} 0 R >>",6+i*2).into_bytes());
        objects.push(stream(content, ""));
    }
    let mut bytes = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![0];
    for (i, object) in objects.iter().enumerate() {
        offsets.push(bytes.len());
        writeln!(bytes, "{} 0 obj", i + 1).unwrap();
        bytes.extend(object);
        bytes.extend(b"\nendobj\n");
    }
    let xref = bytes.len();
    write!(bytes, "xref\n0 {}\n0000000000 65535 f \n", offsets.len()).unwrap();
    for offset in offsets.iter().skip(1) {
        writeln!(bytes, "{offset:010} 00000 n ").unwrap();
    }
    writeln!(
        bytes,
        "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{xref}\n%%EOF",
        offsets.len()
    )
    .unwrap();
    bytes
}

pub const DRAWING: &[u8] = b"1 0 0 rg 60 100 80 120 re f";
pub const IMAGE: &[u8] = b"q 612 0 0 792 0 0 cm /Im Do Q";
pub fn drawing(extra: &str) -> Vec<u8> {
    document(&[(DRAWING, extra)], None)
}
pub fn text(lines: &[&str]) -> Vec<u8> {
    let mut content = String::from("BT /F1 20 Tf 50 720 Td 30 TL\n");
    for line in lines {
        let escaped = line
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        content.push_str(&format!("({escaped}) Tj T* /F1 13 Tf\n"));
    }
    content.push_str("ET");
    document(&[(content.as_bytes(), "")], None)
}
pub fn rgb_image(width: u32, height: u32, rgb: &[u8], mixed: bool) -> Vec<u8> {
    assert_eq!(rgb.len(), width as usize * height as usize * 3);
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(rgb).unwrap();
    let image = stream(
        &encoder.finish().unwrap(),
        &format!(
            "/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode"
        ),
    );
    let mut pages = Vec::new();
    if mixed {
        pages.push((
            b"BT /F1 12 Tf 50 720 Td (Public garden with basil and lavender.) Tj ET".as_slice(),
            "",
        ));
    }
    pages.push((IMAGE, ""));
    document(&pages, Some(image))
}
pub fn broken_image(marker: &str) -> Vec<u8> {
    let image = stream(
        marker.as_bytes(),
        "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode",
    );
    document(&[(IMAGE, "")], Some(image))
}
