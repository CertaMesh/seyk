#![allow(dead_code)]
use std::io::Write;

pub const MARKER: &str = "SEYK_SYNTHETIC_PRIVATE_MARKER_4A_72C10";
pub const PUBLIC: &str = "Public fixture contains a garden with basil and lavender.";

#[derive(Clone, Copy)]
pub enum Kind {
    Plain,
    Tint,
    InvalidUtf8,
    Panic,
    BrokenSecond,
    Encrypted,
    MissingContents,
    MissingArrayContents,
    TruncatedContents,
    BadFlate,
    EmptyPage,
}

fn stream(body: &[u8], extra: &str) -> Vec<u8> {
    let mut out = format!("<< /Length {} {} >>\nstream\n", body.len(), extra).into_bytes();
    out.extend_from_slice(body);
    out.extend_from_slice(b"\nendstream");
    out
}

pub fn pdf(texts: &[&str], kind: Kind) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            (0..texts.len())
                .map(|i| format!("{} 0 R", 5 + i * 2))
                .collect::<Vec<_>>()
                .join(" "),
            texts.len()
        )
        .into_bytes(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
    ];
    let mut function = format!("{{ pop 0 }} % {MARKER}").into_bytes();
    if matches!(kind, Kind::InvalidUtf8) {
        function.push(255);
    }
    objects.push(stream(
        &function,
        "/FunctionType 4 /Domain [0 1] /Range [0 1]",
    ));
    for (i, text) in texts.iter().enumerate() {
        let contents = if i == texts.len() - 1 {
            match kind {
                Kind::MissingContents => "/Contents 999 0 R".into(),
                Kind::MissingArrayContents => format!("/Contents [{} 0 R 999 0 R]", 6 + i * 2),
                Kind::EmptyPage => String::new(),
                _ => format!("/Contents {} 0 R", 6 + i * 2),
            }
        } else {
            format!("/Contents {} 0 R", 6 + i * 2)
        };
        objects.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> /ColorSpace << /CS1 [/Separation /Spot /DeviceGray 4 0 R] >> >> {} >>", contents).into_bytes());
        let prefix = if matches!(kind, Kind::Tint | Kind::InvalidUtf8) {
            "/CS1 cs 0.5 scn "
        } else {
            ""
        };
        let escaped = text
            .replace('\\', "\\\\")
            .replace('(', "\\(")
            .replace(')', "\\)");
        let mut content =
            if matches!(kind, Kind::Panic) || (matches!(kind, Kind::BrokenSecond) && i == 1) {
                format!("BT /F1 12 Tf 72 720 Td /{MARKER} Tj ET")
            } else {
                format!("{prefix}BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET")
            };
        if matches!(kind, Kind::TruncatedContents) && i == texts.len() - 1 {
            content.push_str(" (unterminated");
        }
        // WinAnsi fixture supports Latin-1; ToUnicode-specific tests use a separate fixture.
        let bytes: Vec<u8> = content
            .chars()
            .map(|c| u8::try_from(c as u32).expect("Latin-1 fixture"))
            .collect();
        objects.push(stream(
            &bytes,
            if matches!(kind, Kind::BadFlate) && i == texts.len() - 1 {
                "/Filter /FlateDecode"
            } else {
                ""
            },
        ));
    }
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        writeln!(pdf, "{} 0 obj", i + 1).unwrap();
        pdf.extend_from_slice(obj);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    writeln!(pdf, "xref\n0 {}\n0000000000 65535 f ", objects.len() + 1).unwrap();
    for offset in offsets {
        writeln!(pdf, "{offset:010} 00000 n ").unwrap();
    }
    let encrypt = if matches!(kind, Kind::Encrypted) {
        "/Encrypt << /Filter /Standard /V 1 /R 2 /Length 40 /O () /U () /P -4 >>"
    } else {
        ""
    };
    writeln!(
        pdf,
        "trailer\n<< /Size {} /Root 1 0 R {} >>\nstartxref\n{xref}\n%%EOF",
        objects.len() + 1,
        encrypt
    )
    .unwrap();
    pdf
}

/// Self-made minimal ToUnicode fixtures. No source corpus bytes are used.
/// One map overlaps an unknown FontAwesome glyph; the other has a three-byte
/// bfrange destination, triggering adobe-cmap-parser's length panic.
pub fn malformed_cmap(bad_length: bool) -> Vec<u8> {
    let cmap = if bad_length {
        b"1 beginbfrange <41> <41> <000041> endbfrange".as_slice()
    } else {
        b"1 beginbfchar <41> <0041> endbfchar".as_slice()
    };
    let font = if bad_length {
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>"
    } else {
        "<< /Type /Font /Subtype /Type1 /BaseFont /FontAwesome /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 /UnknownSyntheticGlyph] >> /ToUnicode 6 0 R >>"
    };
    let objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_vec(),
        font.as_bytes().to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>".to_vec(),
        stream(b"BT /F1 12 Tf 72 720 Td (A) Tj ET", ""),
        stream(cmap, ""),
    ];
    let mut bytes = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(bytes.len());
        writeln!(bytes, "{} 0 obj", i + 1).unwrap();
        bytes.extend_from_slice(object);
        bytes.extend_from_slice(b"\nendobj\n");
    }
    let xref = bytes.len();
    writeln!(bytes, "xref\n0 7\n0000000000 65535 f ").unwrap();
    for offset in offsets {
        writeln!(bytes, "{offset:010} 00000 n ").unwrap();
    }
    writeln!(
        bytes,
        "trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    )
    .unwrap();
    bytes
}
