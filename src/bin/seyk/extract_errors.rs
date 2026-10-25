//! Opt-in runtime diagnostics. Keep classification independent of error prose.
use std::fmt;

/// A parsing/selection stage rejected an envelope. Display preserves legacy prose.
#[derive(Debug)]
pub(super) struct InvalidInput(pub String);

impl fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InvalidInput {}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Failure {
    pub exit: u8,
    pub code: &'static str,
}

impl Failure {
    pub fn classify(error: &anyhow::Error) -> Self {
        // Inspect enclosing domain errors first, including unlisted variants.
        // A nested I/O source must not overwrite their more specific meaning.
        if error.downcast_ref::<InvalidInput>().is_some()
            || error.chain().any(|cause| cause.is::<InvalidInput>())
        {
            return Self::new(3, "invalid_input");
        }
        if let Some(core) = error.downcast_ref::<seyk::Error>().or_else(|| {
            error
                .chain()
                .find_map(|cause| cause.downcast_ref::<seyk::Error>())
        }) {
            use seyk::Error::*;
            return match core {
                StaleResource { .. } | ChangedWhileReading { .. } => {
                    Self::new(4, "resource_changed")
                }
                UnsupportedExtraction { .. } | PdfOcrUnavailable { .. } => {
                    Self::new(5, "unsupported_extraction")
                }
                InvalidLocator { .. } | InvalidText { .. } | AudioIntervalTooLong => {
                    Self::new(6, "invalid_evidence")
                }
                Io { .. } => Self::new(7, "io_failed"),
                SecretFile { .. } => Self::new(8, "policy_denied"),
                UnsupportedAudio { .. }
                | PdfParse { .. }
                | PdfPageDecode { .. }
                | ParserPanicked { .. }
                | PdfRender { .. } => Self::new(9, "invalid_media"),
                SttModel { .. } => Self::new(10, "stt_model_unavailable"),
                SpeechToText { .. } => Self::new(11, "transcription_failed"),
                _ => Self::new(1, "unknown_failure"),
            };
        }
        if error.chain().any(|cause| {
            cause.is::<std::io::Error>()
                || cause
                    .downcast_ref::<serde_json::Error>()
                    .is_some_and(serde_json::Error::is_io)
        }) {
            return Self::new(7, "io_failed");
        }
        Self::new(1, "unknown_failure")
    }

    fn new(exit: u8, code: &'static str) -> Self {
        Self { exit, code }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error: {}", self.code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    const SENTINEL: &str = "SYNTHETIC_PRIVATE_SENTINEL";

    #[test]
    fn extract_classifier_table_and_wrapped_errors() {
        use seyk::Error::*;
        let resource: seyk::ResourceId = format!("mem://{SENTINEL}").parse().unwrap();
        let locator = seyk::ResourceLocator::WholeResource;
        let errors = vec![
            (
                StaleResource {
                    resource: resource.clone(),
                    expected: SENTINEL.into(),
                    actual: SENTINEL.into(),
                },
                4,
                "resource_changed",
            ),
            (
                ChangedWhileReading {
                    resource: resource.clone(),
                },
                4,
                "resource_changed",
            ),
            (
                UnsupportedExtraction {
                    resource: resource.clone(),
                    modality: seyk::Modality::Text,
                    locator: locator.clone(),
                },
                5,
                "unsupported_extraction",
            ),
            (
                PdfOcrUnavailable {
                    resource: resource.clone(),
                    page: 1,
                },
                5,
                "unsupported_extraction",
            ),
            (
                InvalidLocator {
                    resource: resource.clone(),
                    locator,
                    reason: SENTINEL,
                },
                6,
                "invalid_evidence",
            ),
            (
                InvalidText {
                    resource: resource.clone(),
                    source: String::from_utf8(vec![255]).unwrap_err().utf8_error(),
                },
                6,
                "invalid_evidence",
            ),
            (AudioIntervalTooLong, 6, "invalid_evidence"),
            (
                Io {
                    path: SENTINEL.into(),
                    source: std::io::Error::other(SENTINEL),
                },
                7,
                "io_failed",
            ),
            (
                SecretFile {
                    path: SENTINEL.into(),
                },
                8,
                "policy_denied",
            ),
            (
                UnsupportedAudio {
                    resource: resource.clone(),
                    reason: SENTINEL,
                },
                9,
                "invalid_media",
            ),
            (
                PdfParse {
                    resource: resource.clone(),
                    reason: SENTINEL,
                },
                9,
                "invalid_media",
            ),
            (
                PdfPageDecode {
                    resource: resource.clone(),
                    reason: SENTINEL,
                },
                9,
                "invalid_media",
            ),
            (
                ParserPanicked {
                    resource: resource.clone(),
                    reason: SENTINEL,
                },
                9,
                "invalid_media",
            ),
            (
                PdfRender {
                    resource,
                    reason: SENTINEL,
                },
                9,
                "invalid_media",
            ),
            (SttModel { reason: SENTINEL }, 10, "stt_model_unavailable"),
            (
                SpeechToText { reason: SENTINEL },
                11,
                "transcription_failed",
            ),
            (Config(SENTINEL.into()), 1, "unknown_failure"),
            (Embedding(SENTINEL.into()), 1, "unknown_failure"),
            (ModelNotFound(SENTINEL.into()), 1, "unknown_failure"),
            (CacheCleanupPending, 1, "unknown_failure"),
        ];
        for (error, exit, code) in errors {
            let error = anyhow::Error::new(error)
                .context(SENTINEL)
                .context(SENTINEL);
            let failure = Failure::classify(&error);
            assert_eq!(failure, Failure::new(exit, code));
            assert_eq!(format!("{failure}\n"), format!("error: {code}\n"));
            assert!(!failure.to_string().contains(SENTINEL));
        }
        let unknown = anyhow::anyhow!(SENTINEL);
        assert_eq!(
            Failure::classify(&unknown),
            Failure::new(1, "unknown_failure")
        );
        let input = Err::<(), _>(std::io::Error::other(SENTINEL))
            .context(InvalidInput(SENTINEL.into()))
            .unwrap_err();
        assert_eq!(Failure::classify(&input), Failure::new(3, "invalid_input"));
    }
}
