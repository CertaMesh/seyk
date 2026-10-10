//! Bounded embedded-text extraction from PDF bytes.
//!
//! Applications with a logger must filter `pdf_extract` to Off, plus
//! `hayro_interpret`, `hayro_syntax` and `hayro_font` with `pdf-render`. This module
//! catches unwinding parser panics with a scoped, thread-local hook guard that
//! delegates unrelated panics to the previous hook. Abort, OOM and
//! parser hangs are not contained. `pdf-render` adds CPU page rendering with
//! bundled fallback fonts and no system-font lookup. No OCR is performed.

use crate::extractor::{ExtractedContent, ExtractionProvenance, Extractor};
use crate::{Error, Modality, ResourceId, ResourceLocator, ResourceMetadata, Result};
use pdf_extract::{Document, PlainTextOutput};
use std::io::Write;

pub const TEXT_VERSION: &str = "pdf-extract-0.12.1;plain-v1";
pub const MAX_PAGES: usize = 512;
pub const MAX_PAGE_TEXT_BYTES: usize = 1024 * 1024;
pub const MAX_DOCUMENT_TEXT_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const PREPARATION_VERSION: &str = "pdf-text-v3-page-isolation;512p;1MiB-page;32MiB-doc";

#[cfg(feature = "pdf-render")]
#[path = "pdf/render.rs"]
pub(crate) mod render;
#[cfg(feature = "pdf-render")]
pub use render::{RENDER_VERSION, render_page};

pub(crate) struct PageText {
    pub page: u32,
    pub text: String,
    pub error: Option<Error>,
}

fn failure(source: &ResourceId, reason: &'static str) -> Error {
    Error::PdfParse {
        resource: source.clone(),
        reason,
    }
}

/// All parser work, including document loading and destruction, stays inside
/// this boundary. Error strings/panic payloads from upstream are never exposed.
fn guarded<T>(source: &ResourceId, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let _silence = panic_hook::Guard::new();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(Error::ParserPanicked {
            resource: source.clone(),
        })
    })
}

fn document(bytes: &[u8], source: &ResourceId) -> Result<Document> {
    let mut doc = Document::load_mem(bytes).map_err(|_| failure(source, "invalid document"))?;
    if doc.is_encrypted() {
        return Err(failure(source, "encrypted documents are unsupported"));
    }
    // Without a logger, log! arguments are not evaluated. Reject invalid Type-4
    // UTF-8 explicitly instead of depending on upstream's log-only unwrap panic.
    for object in doc.objects.values() {
        if let Ok(stream) = object.as_stream()
            && stream
                .dict
                .get(b"FunctionType")
                .and_then(|v| v.as_i64())
                .ok()
                == Some(4)
        {
            let content = stream
                .get_plain_content()
                .map_err(|_| failure(source, "invalid function stream"))?;
            if std::str::from_utf8(&content).is_err() {
                return Err(failure(source, "non-UTF-8 Type-4 function"));
            }
        }
    }
    // Validate the whole resource even when the caller only requests one page.
    // Normalize to precisely the validated bytes so upstream cannot fall back to
    // raw compressed data or silently omit an unresolved reference.
    let numbers = pages(&doc, source)?;
    let page_ids = doc.get_pages();
    for number in numbers {
        let id = page_ids[&number];
        let page = doc
            .get_dictionary(id)
            .map_err(|_| failure(source, "invalid page"))?;
        let mut content = Vec::new();
        if let Ok(contents) = page.get(b"Contents") {
            let (_, resolved) = doc
                .dereference(contents)
                .map_err(|_| failure(source, "unresolved page contents"))?;
            let objects: &[pdf_extract::Object] = match resolved {
                pdf_extract::Object::Array(items) => items,
                pdf_extract::Object::Null => &[],
                pdf_extract::Object::Stream(_) => std::slice::from_ref(contents),
                _ => return Err(failure(source, "page contents must reference streams")),
            };
            for object in objects {
                let (_, object) = doc
                    .dereference(object)
                    .map_err(|_| failure(source, "unresolved page content stream"))?;
                let stream = object
                    .as_stream()
                    .map_err(|_| failure(source, "page content is not a stream"))?;
                content.extend(strict_plain_content(stream, source)?);
                // Match lopdf's page-stream concatenation exactly.
                content.push(b'\n');
            }
        }
        pdf_extract::content::Content::decode_strict(&content)
            .map_err(|_| failure(source, "invalid or incomplete page content"))?;
        let stream = doc.add_object(pdf_extract::Stream::new(
            pdf_extract::Dictionary::new(),
            content,
        ));
        doc.get_object_mut(id)
            .and_then(pdf_extract::Object::as_dict_mut)
            .map_err(|_| failure(source, "invalid page"))?
            .set("Contents", stream);
    }
    Ok(doc)
}

fn strict_plain_content(stream: &pdf_extract::Stream, source: &ResourceId) -> Result<Vec<u8>> {
    // Upstream Flate/LZW helpers log decompression errors and return prefixes.
    // Content filters we cannot verify strictly are explicit unsupported errors.
    if let Ok(params) = stream.dict.get(b"DecodeParms")
        && !matches!(params, pdf_extract::Object::Null)
    {
        return Err(failure(source, "content DecodeParms are unsupported"));
    }
    let mut bytes = stream.content.clone();
    if stream.dict.get(b"Filter").is_err() {
        return Ok(bytes);
    }
    let filters = stream
        .filters()
        .map_err(|_| failure(source, "invalid content filter"))?;
    if filters.is_empty() {
        return Err(failure(source, "empty content filter"));
    }
    for filter in filters {
        if filter != b"FlateDecode" {
            return Err(failure(
                source,
                "content filter lacks strict validation support",
            ));
        }
        let mut decoder = flate2::Decompress::new(true);
        let mut decoded = Vec::new();
        loop {
            let mut chunk = [0u8; 8192];
            let before_in = decoder.total_in();
            let before_out = decoder.total_out();
            let status = decoder
                .decompress(
                    &bytes[before_in as usize..],
                    &mut chunk,
                    flate2::FlushDecompress::None,
                )
                .map_err(|_| failure(source, "invalid compressed page content"))?;
            decoded.extend_from_slice(&chunk[..(decoder.total_out() - before_out) as usize]);
            if status == flate2::Status::StreamEnd {
                if decoder.total_in() as usize != bytes.len() {
                    return Err(failure(source, "trailing compressed page content"));
                }
                break;
            }
            if decoder.total_in() == before_in && decoder.total_out() == before_out {
                return Err(failure(source, "incomplete compressed page content"));
            }
        }
        bytes = decoded;
    }
    Ok(bytes)
}

struct LimitedText {
    bytes: Vec<u8>,
    exceeded: bool,
}
impl Write for LimitedText {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_PAGE_TEXT_BYTES.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(std::io::Error::other("PDF text limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn decode(doc: &Document, source: &ResourceId, page: u32) -> Result<String> {
    let mut text = LimitedText {
        bytes: Vec::new(),
        exceeded: false,
    };
    let result = pdf_extract::output_doc_page(
        doc,
        &mut PlainTextOutput::new(&mut text as &mut dyn Write),
        page,
    );
    if text.exceeded {
        return Err(failure(source, "page text exceeds 1 MiB"));
    }
    result.map_err(|_| failure(source, "page decoding failed"))?;
    String::from_utf8(text.bytes).map_err(|_| failure(source, "decoded text is not UTF-8"))
}

#[path = "pdf/panic_hook.rs"]
mod panic_hook;

fn pages(doc: &Document, source: &ResourceId) -> Result<Vec<u32>> {
    let expected = doc
        .catalog()
        .and_then(|c| c.get(b"Pages"))
        .and_then(|o| o.as_reference())
        .and_then(|id| doc.get_dictionary(id))
        .and_then(|d| d.get(b"Count"))
        .and_then(|o| o.as_i64())
        .map_err(|_| failure(source, "invalid page tree"))?;
    if expected > MAX_PAGES as i64 {
        return Err(failure(source, "document exceeds 512 pages"));
    }
    let pages: Vec<_> = doc.get_pages().into_keys().collect();
    if expected < 0 || pages.len() as i64 != expected {
        return Err(failure(source, "incomplete page tree"));
    }
    if pages.is_empty() {
        return Err(failure(source, "document has no pages"));
    }
    if pages.len() > MAX_PAGES {
        return Err(failure(source, "document exceeds 512 pages"));
    }
    Ok(pages)
}

pub(crate) fn read_pages(bytes: &[u8], source: &ResourceId) -> Result<Vec<PageText>> {
    guarded(source, || {
        let doc = document(bytes, source)?;
        let mut total = 0usize;
        let mut result = Vec::new();
        for page in pages(&doc, source)? {
            let (text, error) = match guarded(source, || decode(&doc, source, page)) {
                Ok(text) => (text, None),
                Err(
                    error @ (Error::ParserPanicked { .. }
                    | Error::PdfParse {
                        reason: "page decoding failed",
                        ..
                    }),
                ) => (String::new(), Some(error)),
                Err(error) => return Err(error),
            };
            total += text.len();
            if total > MAX_DOCUMENT_TEXT_BYTES {
                return Err(failure(source, "document text exceeds 32 MiB"));
            }
            result.push(PageText { page, text, error });
        }
        Ok(result)
    })
}

pub(crate) fn usable(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric) && !text.contains(['\0', '\u{fffd}'])
}

pub(crate) fn read_page(bytes: &[u8], source: &ResourceId, page: u32) -> Result<String> {
    guarded(source, || {
        let doc = document(bytes, source)?;
        if !pages(&doc, source)?.contains(&page) {
            return Err(failure(source, "page is out of bounds"));
        }
        decode(&doc, source, page)
    })
}

pub(crate) fn slice<'a>(
    text: &'a str,
    source: &ResourceId,
    locator: &ResourceLocator,
) -> Result<&'a str> {
    let invalid = |reason| Error::InvalidLocator {
        resource: source.clone(),
        locator: locator.clone(),
        reason,
    };
    let ResourceLocator::PageText {
        byte_start,
        byte_end,
        text_version,
        page_text_hash,
        ..
    } = locator
    else {
        return Err(invalid("expected a PDF page text range"));
    };
    if text_version != TEXT_VERSION {
        return Err(invalid("PDF text version changed; reindex"));
    }
    if blake3::hash(text.as_bytes()).to_hex().as_str() != page_text_hash {
        return Err(invalid("PDF page text changed; reindex"));
    }
    let start =
        usize::try_from(*byte_start).map_err(|_| invalid("byte offset exceeds platform limits"))?;
    let end =
        usize::try_from(*byte_end).map_err(|_| invalid("byte offset exceeds platform limits"))?;
    text.get(start..end)
        .ok_or_else(|| invalid("range must be ordered, in bounds and on UTF-8 boundaries"))
}

#[derive(Debug, Default)]
pub struct PdfTextExtractor;
impl Extractor for PdfTextExtractor {
    fn supports(
        &self,
        _: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> bool {
        metadata.modality == Modality::Document
            && metadata.media_type.as_deref() == Some("application/pdf")
            && matches!(locator, ResourceLocator::PageText { .. })
    }
    fn extract(
        &self,
        bytes: &[u8],
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> Result<ExtractedContent> {
        if !self.supports(source, locator, metadata) {
            return Err(crate::extractor::unsupported(source, locator, metadata));
        }
        let ResourceLocator::PageText { page, .. } = locator else {
            unreachable!()
        };
        let text = read_page(bytes, source, *page)?;
        Ok(ExtractedContent {
            text: slice(&text, source, locator)?.to_owned(),
            source: source.clone(),
            locator: locator.clone(),
            metadata: metadata.clone(),
            provenance: Some(ExtractionProvenance {
                speech: None,
                method: "embedded_pdf_text".into(),
                implementation: TEXT_VERSION.into(),
            }),
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/pdf_fixture.rs"]
pub(crate) mod fixture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_made_cmaps_reproduce_both_upstream_panics() {
        let source = "mem://synthetic/cmap".parse().unwrap();
        for (bad_length, message) in [
            (false, "unexpected entry in unicode map"),
            (true, "bad length of hexstring"),
        ] {
            let bytes = fixture::malformed_cmap(bad_length);
            let doc = document(&bytes, &source).unwrap();
            let _silence = panic_hook::Guard::new();
            let payload =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(&doc, &source, 1)))
                    .unwrap_err();
            let actual = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
            assert_eq!(actual, Some(message));
        }
    }

    #[test]
    fn page_and_document_output_caps_reject_without_truncation() {
        let source = "mem://synthetic/limits".parse().unwrap();
        let oversized = "a".repeat(MAX_PAGE_TEXT_BYTES + 1);
        assert!(matches!(
            read_pages(&fixture::pdf(&[&oversized], fixture::Kind::Plain), &source),
            Err(Error::PdfParse {
                reason: "page text exceeds 1 MiB",
                ..
            })
        ));
        // Reuse one stream reference across physical pages to keep input tiny;
        // decoded output must still count on every page.
        let text = "a".repeat(1024 * 1024 - 1024);
        let mut doc =
            Document::load_mem(&fixture::pdf(&vec!["a"; 33], fixture::Kind::Plain)).unwrap();
        let body = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET").into_bytes();
        doc.objects.insert(
            (6, 0),
            pdf_extract::Object::Stream(pdf_extract::Stream::new(
                pdf_extract::Dictionary::new(),
                body,
            )),
        );
        for id in doc.get_pages().values().copied().collect::<Vec<_>>() {
            doc.get_object_mut(id)
                .unwrap()
                .as_dict_mut()
                .unwrap()
                .set("Contents", (6, 0));
        }
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        assert!(matches!(
            read_pages(&bytes, &source),
            Err(Error::PdfParse {
                reason: "document text exceeds 32 MiB",
                ..
            })
        ));
        let small = fixture::pdf(&vec!["garden basil"; MAX_PAGES], fixture::Kind::Plain);
        assert_eq!(read_pages(&small, &source).unwrap().len(), MAX_PAGES);
    }
    #[test]
    fn flate_requires_stream_end_checksum_and_complete_content() {
        use flate2::{Compression, write::ZlibEncoder};
        let source = "mem://synthetic/flate".parse().unwrap();
        let body = b"BT /F1 12 Tf 72 720 Td (garden basil) Tj ET";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(body).unwrap();
        let compressed = encoder.finish().unwrap();
        for variant in 0..4 {
            let mut bytes = compressed.clone();
            match variant {
                1 => {
                    bytes.truncate(bytes.len() - 2);
                }
                2 => {
                    *bytes.last_mut().unwrap() ^= 1;
                }
                3 => bytes.push(0),
                _ => {}
            }
            let mut doc =
                Document::load_mem(&fixture::pdf(&["placeholder"], fixture::Kind::Plain)).unwrap();
            let stream = doc.get_object_mut((6, 0)).unwrap().as_stream_mut().unwrap();
            stream.set_content(bytes);
            stream.dict.set("Filter", "FlateDecode");
            let mut pdf = Vec::new();
            doc.save_to(&mut pdf).unwrap();
            let result = read_page(&pdf, &source, 1);
            if variant == 0 {
                assert!(result.unwrap().contains("garden basil"));
            } else {
                assert!(matches!(result, Err(Error::PdfParse { .. })));
            }
        }
    }

    #[test]
    fn selected_page_validation_rejects_corruption_elsewhere() {
        let source = "mem://synthetic/corrupt".parse().unwrap();
        for kind in [
            fixture::Kind::MissingContents,
            fixture::Kind::MissingArrayContents,
            fixture::Kind::TruncatedContents,
            fixture::Kind::BadFlate,
        ] {
            let bytes = fixture::pdf(&["valid first page", "valid prefix"], kind);
            assert!(matches!(
                read_page(&bytes, &source, 1),
                Err(Error::PdfParse { .. })
            ));
        }
    }
}
