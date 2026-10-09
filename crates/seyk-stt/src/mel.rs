//! OpenAI Whisper mel80 contract; separate from Gemma audio preparation.
use rustfft::{FftPlanner, num_complex::Complex};
use seyk_core::{Error, Result};
const FILTERS: &[u8] = include_bytes!("../assets/mel-80.f32le");
pub(crate) fn features(samples: &[f32]) -> Result<Vec<f32>> {
    if samples.len() > 480_000 {
        return Err(Error::AudioIntervalTooLong);
    }
    if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
        return Err(Error::SpeechToText {
            reason: "invalid canonical PCM",
        });
    }
    let mut audio = vec![0.0f32; 480_000];
    audio[..samples.len()].copy_from_slice(samples);
    let window: Vec<f32> = (0..400)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 400.0).cos()) as f32)
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(400);
    let mut frame = vec![Complex::default(); 400];
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut power = [0.0f32; 201];
    let filters: Vec<f32> = FILTERS
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let mut mel = vec![0.0f32; 80 * 3000];
    for t in 0..3000 {
        for i in 0..400 {
            let at = t as isize * 160 + i as isize - 200;
            // Reflect the zero-padded 30 s signal, not the selected interior edge.
            let at = if at < 0 {
                -at
            } else if at >= 480_000 {
                959_998 - at
            } else {
                at
            } as usize;
            frame[i] = Complex::new(audio[at] * window[i], 0.0);
        }
        fft.process_with_scratch(&mut frame, &mut scratch);
        for (v, x) in power.iter_mut().zip(&frame) {
            *v = x.norm_sqr();
        }
        for band in 0..80 {
            let sum: f32 = filters[band * 201..(band + 1) * 201]
                .iter()
                .zip(power)
                .map(|(a, b)| a * b)
                .sum();
            if !sum.is_finite() {
                return Err(Error::SpeechToText {
                    reason: "nonfinite mel energy",
                });
            }
            mel[band * 3000 + t] = sum.max(1e-10).log10();
        }
    }
    let floor = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for x in &mut mel {
        *x = (x.max(floor) + 4.0) / 4.0;
    }
    Ok(mel)
}
