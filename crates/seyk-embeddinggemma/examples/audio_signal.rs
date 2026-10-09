//! Regenerate the small, synthetic parity input (no model files required).
#[path = "../tests/support/audio_signal.rs"]
mod fixture;
fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).expect("output .f32 path");
    let samples = std::env::args()
        .nth(2)
        .map(|v| v.parse().expect("sample count"))
        .unwrap_or(16_000);
    std::fs::write(
        path,
        fixture::signal_len(samples)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
