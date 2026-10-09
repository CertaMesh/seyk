/// One second of deterministic chirp plus tones, quantized to signed 16-bit PCM.
/// No speech synthesis and no source corpus audio.
pub fn signal() -> Vec<f32> {
    (0..16_000)
        .map(|i| {
            let t = i as f64 / 16_000.0;
            let phase = std::f64::consts::TAU;
            let sample = 0.25 * (phase * (180.0 * t + 1700.0 * t * t)).sin()
                + 0.125 * (phase * 440.0 * t).sin()
                + 0.0625 * (phase * 2300.0 * t).sin();
            (sample * 32767.0).round() as i16 as f32 / 32768.0
        })
        .collect()
}

/// Repeat the same deterministic source to exercise tail and full-window shapes.
pub fn signal_len(samples: usize) -> Vec<f32> {
    signal().into_iter().cycle().take(samples).collect()
}
