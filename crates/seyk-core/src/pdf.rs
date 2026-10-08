//! Bounded embedded-text extraction from PDF bytes.
//!
//! Applications with a logger must filter `pdf_extract` to Off. This module
//! catches unwinding parser panics without exposing their payload, but installs
//! no panic hook. The caller's hook must also avoid payloads. Abort, OOM and
//! parser hangs are not contained. No OCR or rendering is performed.

use crate::extractor::{ExtractedContent, ExtractionProvenance, Extractor};
use crate::{Error, Modality, ResourceId, ResourceLocator, ResourceMetadata, Result};
use pdf_extract::{Document, PlainTextOutput};
use std::io::Write;

pub const TEXT_VERSION: &str = "pdf-extract-0.12.1;plain-v1";
pub const MAX_PAGES: usize = 512;
pub const MAX_PAGE_TEXT_BYTES: usize = 1024 * 1024;
pub const MAX_DOCUMENT_TEXT_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const PREPARATION_VERSION: &str = "pdf-text-v1;512p;1MiB-page;32MiB-doc";

pub(crate) struct PageText {
    pub page: u32,
    pub text: String,
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
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(Error::ParserPanicked {
            resource: source.clone(),
        })
    })
}

fn document(bytes: &[u8], source: &ResourceId) -> Result<Document> {
    let doc = Document::load_mem(bytes).map_err(|_| failure(source, "invalid document"))?;
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
    Ok(doc)
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
            let text = decode(&doc, source, page)?;
            total += text.len();
            if total > MAX_DOCUMENT_TEXT_BYTES {
                return Err(failure(source, "document text exceeds 32 MiB"));
            }
            result.push(PageText { page, text });
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
                method: "embedded_pdf_text".into(),
                implementation: TEXT_VERSION.into(),
            }),
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/pdf_fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::*;

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
}
