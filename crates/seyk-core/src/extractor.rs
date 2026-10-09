//! On-demand readable evidence, independent of models and caches.

use crate::{
    Error, Modality, ResourceId, ResourceLocator, ResourceMetadata, ResourceProvider, Result,
};
use serde::{Deserialize, Serialize};

/// Readable content from a selected location in a verified resource snapshot.
///
/// Unstable: may change before 1.0 (a range/streaming method is planned for audio and video).
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// How readable text was derived; absent for exact raw text slices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ExtractionProvenance>,
    pub text: String,
    pub source: ResourceId,
    pub locator: ResourceLocator,
    /// Includes the BLAKE3 hash of the entire source resource, not just `text`.
    pub metadata: ResourceMetadata,
}

/// Provenance of derived readable evidence, not a claim about transcription accuracy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionProvenance {
    pub method: String,
    pub implementation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech: Option<SpeechProvenance>,
}

/// Bounds identify model input, not word alignment or transcription accuracy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechProvenance {
    pub fidelity: String,
    pub requested_locator: ResourceLocator,
    pub sample_start: u64,
    pub sample_end: u64,
    pub sample_rate: u32,
    pub manifest_sha256: String,
    /// None means automatic language detection was requested.
    pub language_requested: Option<String>,
    /// None for no_speech; never invent a language for silence.
    pub language_detected: Option<String>,
    pub no_speech: bool,
}

impl std::fmt::Debug for ExtractedContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractedContent")
            .field("text", &"[redacted]")
            .field("source", &self.source)
            .field("locator", &self.locator)
            .field("provenance", &self.provenance)
            .finish_non_exhaustive()
    }
}

/// Convert verified resource bytes into readable evidence.
///
/// Use [`extract_resource`] or [`crate::SearchEngine::extract`] for provider-backed
/// extraction. Direct calls to `extract` must supply bytes already verified
/// against `metadata.content_hash`. The central gateway verifies before calling
/// any implementation. Future range/streaming extractors need a separate method.
///
/// Unstable: may change before 1.0 (a range/streaming method is planned for audio and video).
pub trait Extractor: Send + Sync {
    fn supports(
        &self,
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> bool;
    fn extract(
        &self,
        bytes: &[u8],
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> Result<ExtractedContent>;
}

/// Exact UTF-8 text ranges, preserving whitespace, Markdown and source code.
#[derive(Debug, Default)]
pub struct TextExtractor;

impl Extractor for TextExtractor {
    fn supports(
        &self,
        _source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> bool {
        metadata.modality == Modality::Text && matches!(locator, ResourceLocator::Text { .. })
    }

    fn extract(
        &self,
        bytes: &[u8],
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
    ) -> Result<ExtractedContent> {
        if !self.supports(source, locator, metadata) {
            return Err(unsupported(source, locator, metadata));
        }
        Ok(ExtractedContent {
            provenance: None,
            text: text_range(bytes, source, locator)?.to_owned(),
            source: source.clone(),
            locator: locator.clone(),
            metadata: metadata.clone(),
        })
    }
}

/// Read and verify a full resource before dispatching to an extractor.
///
/// This model-free entry point uses only the supplied provider. It does not
/// establish authorization or authenticate a caller-supplied content hash.
///
/// Unstable: may change before 1.0 (a range/streaming method is planned for audio and video).
pub fn extract_resource(
    provider: &dyn ResourceProvider,
    extractor: &dyn Extractor,
    source: &ResourceId,
    locator: &ResourceLocator,
    metadata: &ResourceMetadata,
) -> Result<ExtractedContent> {
    #[cfg(feature = "audio")]
    if metadata.modality == Modality::Audio {
        // The byte-based extractor contract must never load a long audio file.
        crate::audio::extract_samples(provider, source, locator, metadata, 7200)?;
        return Err(unsupported(source, locator, metadata));
    }
    if !extractor.supports(source, locator, metadata) {
        return Err(unsupported(source, locator, metadata));
    }
    let read = ResourceBytes::read(provider, source)?;
    extractor.extract(read.verify(source, metadata)?, source, locator, metadata)
}

pub(crate) fn unsupported(
    source: &ResourceId,
    locator: &ResourceLocator,
    metadata: &ResourceMetadata,
) -> Error {
    if metadata.modality == Modality::Document
        && metadata.media_type.as_deref() == Some("application/pdf")
        && let ResourceLocator::Page { page } = locator
    {
        return Error::PdfOcrUnavailable {
            resource: source.clone(),
            page: *page,
        };
    }
    Error::UnsupportedExtraction {
        resource: source.clone(),
        modality: metadata.modality,
        locator: locator.clone(),
    }
}

/// A single provider read and its digest; previews retain this only per search.
pub(crate) struct ResourceBytes {
    bytes: Vec<u8>,
    digest: String,
}

impl ResourceBytes {
    pub(crate) fn read(provider: &dyn ResourceProvider, source: &ResourceId) -> Result<Self> {
        let bytes = provider.read(source)?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        Ok(Self { bytes, digest })
    }

    pub(crate) fn verify(&self, source: &ResourceId, metadata: &ResourceMetadata) -> Result<&[u8]> {
        if self.digest != metadata.content_hash {
            return Err(Error::StaleResource {
                resource: source.clone(),
                expected: metadata.content_hash.clone(),
                actual: self.digest.clone(),
            });
        }
        Ok(&self.bytes)
    }
}

pub(crate) fn text_range<'a>(
    bytes: &'a [u8],
    source: &ResourceId,
    locator: &ResourceLocator,
) -> Result<&'a str> {
    let invalid = |reason| Error::InvalidLocator {
        resource: source.clone(),
        locator: locator.clone(),
        reason,
    };
    let ResourceLocator::Text {
        byte_start,
        byte_end,
        ..
    } = locator
    else {
        return Err(invalid("expected a text byte range"));
    };
    let start =
        usize::try_from(*byte_start).map_err(|_| invalid("byte offset exceeds platform limits"))?;
    let end =
        usize::try_from(*byte_end).map_err(|_| invalid("byte offset exceeds platform limits"))?;
    let text = std::str::from_utf8(bytes).map_err(|error| Error::InvalidText {
        resource: source.clone(),
        source: error,
    })?;
    text.get(start..end).ok_or_else(|| {
        invalid("range must be ordered, in bounds, and on UTF-8 character boundaries")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locator(start: u64, end: u64) -> ResourceLocator {
        ResourceLocator::Text {
            byte_start: start,
            byte_end: end,
            line_start: None,
            line_end: None,
        }
    }

    #[test]
    fn scanned_page_extraction_has_a_typed_model_free_error() {
        let source: ResourceId = "mem://scanned.pdf".parse().unwrap();
        let metadata = ResourceMetadata {
            media_type: Some("application/pdf".into()),
            modality: Modality::Document,
            size_bytes: 0,
            modified_ns: None,
            content_hash: String::new(),
        };
        let error = TextExtractor
            .extract(&[], &source, &ResourceLocator::Page { page: 2 }, &metadata)
            .unwrap_err();
        assert!(matches!(error, Error::PdfOcrUnavailable { page: 2, .. }));
        assert!(
            error
                .to_string()
                .contains("readable text for scanned pages needs OCR, which is not available yet")
        );
        let mut other = metadata;
        other.media_type = Some("application/other".into());
        assert!(matches!(
            unsupported(&source, &ResourceLocator::Page { page: 2 }, &other),
            Error::UnsupportedExtraction { .. }
        ));
    }

    #[test]
    fn text_is_exact_and_ranges_are_checked() {
        let source = "mem://notes/example".parse().unwrap();
        for text in [
            "# Title\r\n\n**文書** [link](url)\n",
            "  café\t🙂\r\n",
            "fn main() {\n    // café\n}\n",
        ] {
            let bytes = text.as_bytes();
            let metadata = ResourceMetadata {
                media_type: None,
                modality: Modality::Text,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                content_hash: blake3::hash(bytes).to_hex().to_string(),
            };
            let range = locator(0, bytes.len() as u64);
            let content = TextExtractor
                .extract(bytes, &source, &range, &metadata)
                .unwrap();
            assert_eq!(content.text.as_bytes(), bytes);
            assert_eq!(content.source, source);
            assert_eq!(content.locator, range);
            assert_eq!(content.metadata, metadata);
            assert_eq!(
                text_range(
                    bytes,
                    &source,
                    &locator(bytes.len() as u64, bytes.len() as u64)
                )
                .unwrap(),
                ""
            );
            for range in [locator(2, 1), locator(0, u64::MAX)] {
                assert!(matches!(
                    text_range(bytes, &source, &range),
                    Err(Error::InvalidLocator { .. })
                ));
            }
        }
        assert_eq!(
            text_range("a🙂b".as_bytes(), &source, &locator(1, 5)).unwrap(),
            "🙂"
        );
        for range in [locator(2, 5), locator(1, 4), locator(2, 2)] {
            assert!(matches!(
                text_range("a🙂b".as_bytes(), &source, &range),
                Err(Error::InvalidLocator { .. })
            ));
        }
        assert!(matches!(
            text_range(&[255], &source, &locator(0, 1)),
            Err(Error::InvalidText { .. })
        ));
    }
    struct Bytes(Vec<u8>);
    impl ResourceProvider for Bytes {
        fn list(&self, _: &crate::ResourceScope) -> Result<Vec<crate::ProvidedResource>> {
            panic!("must not enumerate")
        }
        fn read(&self, _: &ResourceId) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }
        fn version(&self, _: &ResourceId) -> Result<crate::ResourceVersion> {
            panic!("must hash actual bytes")
        }
    }
    struct NeverCalled;
    impl Extractor for NeverCalled {
        fn supports(&self, _: &ResourceId, _: &ResourceLocator, _: &ResourceMetadata) -> bool {
            true
        }
        fn extract(
            &self,
            _: &[u8],
            _: &ResourceId,
            _: &ResourceLocator,
            _: &ResourceMetadata,
        ) -> Result<ExtractedContent> {
            panic!("unverified bytes reached extractor")
        }
    }
    #[test]
    fn gateway_checks_hash_before_dispatch_and_rejects_unsupported() {
        let source = "mem://notes/entry".parse().unwrap();
        let provider = Bytes(b"hello".to_vec());
        let mut metadata = ResourceMetadata {
            media_type: None,
            modality: Modality::Text,
            size_bytes: 5,
            modified_ns: None,
            content_hash: blake3::hash(b"hello").to_hex().to_string(),
        };
        assert_eq!(
            extract_resource(
                &provider,
                &TextExtractor,
                &source,
                &locator(1, 4),
                &metadata
            )
            .unwrap()
            .text,
            "ell"
        );
        metadata.content_hash = blake3::hash(b"other").to_hex().to_string();
        assert!(matches!(
            extract_resource(&provider, &NeverCalled, &source, &locator(0, 5), &metadata),
            Err(Error::StaleResource { .. })
        ));
        for range in [
            ResourceLocator::WholeResource,
            ResourceLocator::Page { page: 1 },
            ResourceLocator::Time {
                start_ms: 0,
                end_ms: 1,
            },
            ResourceLocator::ImageRegion {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
        ] {
            assert!(matches!(
                extract_resource(&provider, &TextExtractor, &source, &range, &metadata),
                Err(Error::UnsupportedExtraction { .. })
            ));
        }
        for modality in [Modality::Image, Modality::Audio, Modality::Video] {
            metadata.modality = modality;
            assert!(matches!(
                extract_resource(
                    &provider,
                    &TextExtractor,
                    &source,
                    &locator(0, 5),
                    &metadata
                ),
                Err(Error::UnsupportedExtraction { .. })
            ));
        }
    }
}
