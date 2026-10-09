//! Explicit audio extraction; deliberately separate from generic preview dispatch.
use crate::extractor::{ExtractionProvenance, SpeechProvenance};
use crate::{
    ExtractedContent, ResourceId, ResourceLocator, ResourceMetadata, ResourceProvider, Result,
};
use std::sync::Arc;

/// A derived transcript. Intentionally has no Debug implementation: text is content.
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub no_speech: bool,
}

/// One bounded mono 16 kHz interval, already verified against the selected hit.
/// Implementations must stay offline and must not log or persist content/tokens.
pub trait SpeechToText: Send + Sync {
    fn implementation(&self) -> &'static str;
    fn manifest_sha256(&self) -> String;
    /// `language = None` requests detection. No language is returned for no_speech.
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript>;
}

/// Verified source slicing followed by on-demand STT; no embedding or cache access.
#[derive(Clone)]
pub struct AudioExtractor {
    speech: Arc<dyn SpeechToText>,
    language: Option<String>,
}
impl AudioExtractor {
    pub fn new(speech: impl SpeechToText + 'static) -> Self {
        Self {
            speech: Arc::new(speech),
            language: None,
        }
    }
    /// Explicit language code; otherwise detect from the selected interval only.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    pub fn extract(
        &self,
        provider: &dyn ResourceProvider,
        source: &ResourceId,
        locator: &ResourceLocator,
        metadata: &ResourceMetadata,
        max_duration_secs: u64,
    ) -> Result<ExtractedContent> {
        // Must precede even lazy model loading. No unverified source reaches STT.
        let samples =
            super::extract_samples(provider, source, locator, metadata, max_duration_secs)?;
        let ResourceLocator::Time { start_ms, .. } = locator else {
            unreachable!("slice validates locator")
        };
        let sample_start = start_ms * 16; // checked by slice_verified
        let sample_end = sample_start + samples.len() as u64;
        let result = self.speech.transcribe(&samples, self.language.as_deref())?;
        let (text, language_detected) = if result.no_speech {
            (String::new(), None)
        } else {
            (result.text, result.language)
        };
        Ok(ExtractedContent {
            text,
            source: source.clone(),
            locator: locator.clone(),
            metadata: metadata.clone(),
            provenance: Some(ExtractionProvenance {
                method: "speech_to_text".into(),
                implementation: self.speech.implementation().into(),
                speech: Some(SpeechProvenance {
                    fidelity: "machine_transcription".into(),
                    requested_locator: locator.clone(),
                    sample_start,
                    sample_end,
                    sample_rate: 16_000,
                    manifest_sha256: self.speech.manifest_sha256(),
                    language_requested: self.language.clone(),
                    language_detected,
                    no_speech: result.no_speech,
                }),
            }),
        })
    }
}
