//! One transient document and one bounded raster per render call.
use super::{document, guarded, pages};
use crate::{Error, ResourceId, Result};
use hayro::{RenderSettings, hayro_interpret::InterpreterSettings, hayro_syntax::Pdf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// The encoded image bytes, not this label alone, identify an embedding input.
/// Change this when the renderer, fonts, raster dimensions or encoding changes.
pub const RENDER_VERSION: &str =
    "hayro-0.5.0;foxit-bundled;crop-rotate;white;edge-2048;uniform-8-0.01pct-v1;vello-png-v1";
const LONG_EDGE: u16 = 2048;

/// Render a physical PDF page to PNG using the same policy as scan retrieval.
///
/// Validates all content streams before returning any page. Page numbers are
/// one-based. Rendering uses only the pinned CPU renderer and bundled fonts;
/// no network, system fonts, OCR or persistent images are involved. See the
/// crate-level logging/panic containment requirements. Hayro can recover some
/// malformed image/font data by omitting it; rendering is not PDF conformance
/// validation. Content streams are still strictly validated for the whole PDF.
/// Pixels and parser allocations are transient; the viewport cap does not
/// sandbox malicious decompression, aborts, OOM or hangs.
pub fn render_page(bytes: &[u8], source: &ResourceId, page: u32) -> Result<Vec<u8>> {
    guarded(source, || PageRenderer::new(bytes, source)?.page(page))
}

/// Reused only while processing one resource. The engine must release each PNG
/// after its bounded embedding batch; no all-corpus raster queue is needed.
pub(crate) struct PageRenderer {
    pdf: Pdf,
    source: ResourceId,
}

impl PageRenderer {
    pub(crate) fn new(bytes: &[u8], source: &ResourceId) -> Result<Self> {
        guarded(source, || {
            // Hayro can recover malformed inputs. Preserve the strict whole-
            // resource validation contract before preparing transient renderer bytes.
            let mut validated = document(bytes, source)?;
            let expected = pages(&validated, source)?.len();
            // Supply validated streams even for empty pages. Hayro otherwise
            // omits pages with absent Contents. Resource bytes stay unchanged.
            let mut normalized = Vec::new();
            validated
                .save_to(&mut normalized)
                .map_err(|_| Error::PdfRender {
                    resource: source.clone(),
                    reason: "validated document could not be prepared for rendering",
                })?;
            drop(validated);
            let pdf = Pdf::new(Arc::new(normalized)).map_err(|_| Error::PdfRender {
                resource: source.clone(),
                reason: "renderer could not load document",
            })?;
            if pdf.pages().len() != expected {
                return Err(Error::PdfRender {
                    resource: source.clone(),
                    reason: "renderer page count disagrees with validated document",
                });
            }
            Ok(Self {
                pdf,
                source: source.clone(),
            })
        })
    }

    pub(crate) fn page(&self, number: u32) -> Result<Vec<u8>> {
        self.page_with_settings(number, InterpreterSettings::default())
    }

    fn page_with_settings(
        &self,
        number: u32,
        mut settings: InterpreterSettings,
    ) -> Result<Vec<u8>> {
        guarded(&self.source, || {
            let failure = |reason| Error::PdfRender {
                resource: self.source.clone(),
                reason,
            };
            let index = number
                .checked_sub(1)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n < self.pdf.pages().len())
                .ok_or_else(|| failure("page is out of bounds"))?;
            let page = &self.pdf.pages()[index];
            let crop = page.intersected_crop_box();
            if !crop.width().is_finite()
                || !crop.height().is_finite()
                || crop.width() <= 0.0
                || crop.height() <= 0.0
            {
                return Err(failure("page crop must have a finite positive area"));
            }
            let (width, height) = page.render_dimensions();
            if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
                return Err(failure("page dimensions must be finite and positive"));
            }
            let scale = f32::from(LONG_EDGE) / width.max(height);
            let (width, height) = ((width * scale).floor(), (height * scale).floor());
            if !(1.0..=f32::from(LONG_EDGE)).contains(&width)
                || !(1.0..=f32::from(LONG_EDGE)).contains(&height)
            {
                return Err(failure("page aspect ratio exceeds raster limits"));
            }
            let warned = Arc::new(AtomicBool::new(false));
            let sink = warned.clone();
            settings.warning_sink = Arc::new(move |_| sink.store(true, Ordering::Relaxed));
            let pixmap = hayro::render(
                page,
                &settings,
                &RenderSettings {
                    x_scale: scale,
                    y_scale: scale,
                    width: Some(width as u16),
                    height: Some(height as u16),
                    bg_color: hayro::vello_cpu::color::palette::css::WHITE,
                },
            );
            if warned.load(Ordering::Relaxed) {
                return Err(failure("unsupported font or undecodable image"));
            }
            if has_image_xobjects(page.resources()) && near_uniform(pixmap.data_as_u8_slice()) {
                return Err(failure("scanned page could not be rendered"));
            }
            pixmap
                .into_png()
                .map_err(|_| failure("PNG encoding failed"))
        })
    }
}

// Conservatively inspect inherited and nested form resources. Unknown/cyclic
// XObjects count as potentially containing an image; never recurse unboundedly.
fn has_image_xobjects(resources: &hayro::hayro_syntax::page::Resources<'_>) -> bool {
    let mut current = Some(resources);
    while let Some(resource) = current {
        if image_dictionary(&resource.x_objects, 0) {
            return true;
        }
        current = resource.parent();
    }
    false
}

fn image_dictionary(dict: &hayro::hayro_syntax::object::Dict<'_>, depth: usize) -> bool {
    use hayro::hayro_syntax::object::{Dict, Name, Stream};
    dict.keys().any(|key| {
        if depth >= 16 {
            return true;
        }
        let Some(stream) = dict.get::<Stream<'_>>(key) else {
            return true;
        };
        let dictionary = stream.dict();
        match dictionary
            .get::<Name<'_>>(b"Subtype".as_slice())
            .as_ref()
            .map(|n| n.as_str())
        {
            Some("Image") => true,
            Some("Form") => dictionary
                .get::<Dict<'_>>(b"Resources".as_slice())
                .and_then(|r| r.get::<Dict<'_>>(b"XObject".as_slice()))
                .is_some_and(|d| image_dictionary(&d, depth + 1)),
            _ => true,
        }
    })
}

fn near_uniform(rgba: &[u8]) -> bool {
    let mut low = [255u8; 3];
    let mut high = [0u8; 3];
    let mut nonwhite = 0usize;
    for pixel in rgba.as_chunks::<4>().0.iter() {
        for channel in 0..3 {
            low[channel] = low[channel].min(pixel[channel]);
            high[channel] = high[channel].max(pixel[channel]);
        }
        nonwhite += usize::from(pixel[..3].iter().any(|v| *v < 247));
    }
    (0..3).all(|i| high[i].saturating_sub(low[i]) <= 8) || nonwhite <= (rgba.len() / 4) / 10_000
}

#[cfg(test)]
#[path = "../../tests/support/scan_fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use hayro::vello_cpu::Pixmap;

    fn source() -> ResourceId {
        "mem://synthetic/render".parse().unwrap()
    }

    #[test]
    fn crop_rotation_white_background_and_encoding_are_deterministic() {
        for (extra, expected_size, red_point, digest) in [
            (
                "",
                (1582, 2048),
                (250, 1600),
                "2205640e0711007be8194b79c8310430a13f0311fa51e330de870058378c87ec",
            ),
            (
                "/Rotate 90",
                (2048, 1582),
                (400, 250),
                "f8bc63f5cd95aa694efc168c6d666501fd7c60d5fd10df0f52117d60ea86fd8a",
            ),
            (
                "/CropBox [40 60 400 600]",
                (1365, 2048),
                (200, 1600),
                "04e4615cc61a271a5c53f780a618ba6e0310ed991b1cca2c632ce9630f5a2f68",
            ),
        ] {
            let bytes = fixture::drawing(extra);
            let png = render_page(&bytes, &source(), 1).unwrap();
            assert_eq!(png, render_page(&bytes, &source(), 1).unwrap());
            let pixmap = Pixmap::from_png(png.as_slice()).unwrap();
            assert_eq!((pixmap.width(), pixmap.height()), expected_size);
            let rgba = pixmap.data_as_u8_slice();
            assert_eq!(&rgba[..4], &[255, 255, 255, 255]);
            assert!(rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
            let offset = (red_point.1 * usize::from(pixmap.width()) + red_point.0) * 4;
            assert_eq!(&rgba[offset..offset + 4], &[255, 0, 0, 255]);
            assert_eq!(
                blake3::hash(&png).to_hex().as_str(),
                digest,
                "bump RENDER_VERSION when changing raster/encoding goldens"
            );
        }
    }

    #[test]
    fn renderer_unwind_is_contained_without_retaining_its_payload() {
        // Exercise an unwind from within Hayro via its font callback, using the
        // same guarded render path as production. No global panic hook changes.
        let renderer = PageRenderer::new(&fixture::text(&["public text"]), &source()).unwrap();
        let settings = InterpreterSettings {
            font_resolver: Arc::new(|_| panic!("{}", crate::pdf::fixture::MARKER)),
            ..Default::default()
        };
        let result = renderer.page_with_settings(1, settings);
        assert!(matches!(result, Err(Error::ParserPanicked { .. })));
        assert!(
            !result
                .err()
                .unwrap()
                .to_string()
                .contains(crate::pdf::fixture::MARKER)
        );
    }

    #[test]
    fn a_legitimately_empty_page_keeps_its_physical_number() {
        use crate::pdf::fixture::{Kind, pdf};
        let png = render_page(&pdf(&["valid", ""], Kind::EmptyPage), &source(), 2).unwrap();
        let pixels = Pixmap::from_png(png.as_slice()).unwrap();
        assert!(pixels.data_as_u8_slice().iter().all(|b| *b == 255));
    }

    #[test]
    fn corrupted_stream_anywhere_rejects_selected_page_before_rendering() {
        use crate::pdf::fixture::{Kind, pdf};
        for kind in [
            Kind::MissingContents,
            Kind::MissingArrayContents,
            Kind::TruncatedContents,
            Kind::BadFlate,
        ] {
            assert!(matches!(
                render_page(&pdf(&["valid", "broken"], kind), &source(), 1),
                Err(Error::PdfParse { .. })
            ));
        }
    }

    #[test]
    fn corrupt_image_omission_is_a_sanitized_render_diagnostic() {
        let broken = fixture::broken_image(crate::pdf::fixture::MARKER);
        let error = render_page(&broken, &source(), 1)
            .expect_err("corrupt image must not become a blank scan hit");
        assert!(matches!(
            error,
            Error::PdfRender {
                reason: "scanned page could not be rendered",
                ..
            }
        ));
        assert!(!error.to_string().contains(crate::pdf::fixture::MARKER));
    }

    #[test]
    fn dimensions_and_page_range_are_explicit_errors() {
        let bytes = fixture::drawing("");
        for page in [0, 2, u32::MAX] {
            assert!(matches!(
                render_page(&bytes, &source(), page),
                Err(Error::PdfRender {
                    reason: "page is out of bounds",
                    ..
                })
            ));
        }
        let bad = fixture::drawing("/MediaBox [0 0 0 0]");
        assert!(matches!(
            render_page(&bad, &source(), 1),
            Err(Error::PdfRender { .. })
        ));
    }
}
