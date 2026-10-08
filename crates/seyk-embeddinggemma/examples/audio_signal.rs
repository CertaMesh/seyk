//! Regenerate the small, synthetic parity input (no model files required).
#[path = "../tests/support/audio_signal.rs"]
mod fixture;
fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).expect("output .f32 path");
    std::fs::write(
        path,
        fixture::signal()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
