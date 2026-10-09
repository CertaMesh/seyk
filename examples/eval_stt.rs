//! Local-only synthetic speech evaluation. Emits aggregate metrics, never transcripts.
#[cfg(feature = "audio-stt")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seyk::{
        FsProvider, Modality, ResourceId, ResourceLocator, ResourceMetadata, audio::AudioExtractor,
    };
    use std::{collections::BTreeMap, path::PathBuf};
    #[derive(serde::Deserialize)]
    struct Clip {
        file: PathBuf,
        language: String,
        text: String,
    }
    #[derive(Default, serde::Serialize)]
    struct Metrics {
        clips: usize,
        words: usize,
        word_errors: usize,
        characters: usize,
        character_errors: usize,
        language_matches: usize,
        no_speech: usize,
    }
    fn distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, x) in a.iter().enumerate() {
            let mut diagonal = row[0];
            row[0] = i + 1;
            for (j, y) in b.iter().enumerate() {
                let old = row[j + 1];
                row[j + 1] = (diagonal + usize::from(x != y))
                    .min(row[j] + 1)
                    .min(old + 1);
                diagonal = old;
            }
        }
        row[b.len()]
    }
    fn normalize(s: &str) -> String {
        s.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: eval_stt <exact-model-directory> <local-clip-manifest.json>".into());
    }
    let clips: Vec<Clip> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let extractor = AudioExtractor::new(seyk::stt::Whisper::load(&args[1])?);
    let mut scores: BTreeMap<String, Metrics> = BTreeMap::new();
    for clip in clips {
        let path = std::fs::canonicalize(clip.file)?;
        let bytes = std::fs::read(&path)?;
        // Generator writes canonical mono PCM16 WAVs with a 44-byte header.
        if bytes.len() < 44
            || &bytes[36..40] != b"data"
            || bytes[22..24] != 1u16.to_le_bytes()
            || bytes[24..28] != 16000u32.to_le_bytes()
            || bytes[34..36] != 16u16.to_le_bytes()
        {
            return Err("expected canonical WAV".into());
        }
        let samples = (bytes.len() - 44) / 2;
        let metadata = ResourceMetadata {
            media_type: Some("audio/wav".into()),
            modality: Modality::Audio,
            size_bytes: bytes.len() as u64,
            modified_ns: None,
            content_hash: blake3::hash(&bytes).to_hex().to_string(),
        };
        let result = extractor.extract(
            &FsProvider,
            &ResourceId::from_path(path)?,
            &ResourceLocator::Time {
                start_ms: 0,
                end_ms: (samples as u64).div_ceil(16),
            },
            &metadata,
            7200,
        )?;
        let reference = normalize(&clip.text);
        let hypothesis = normalize(&result.text);
        let words: Vec<_> = reference.split_whitespace().collect();
        let actual: Vec<_> = hypothesis.split_whitespace().collect();
        let chars: Vec<_> = reference.chars().collect();
        let actual_chars: Vec<_> = hypothesis.chars().collect();
        let score = scores.entry(clip.language.clone()).or_default();
        score.clips += 1;
        score.words += words.len();
        score.word_errors += distance(&words, &actual);
        score.characters += chars.len();
        score.character_errors += distance(&chars, &actual_chars);
        let provenance = result.provenance.unwrap().speech.unwrap();
        score.language_matches +=
            usize::from(provenance.language_detected.as_deref() == Some(&clip.language));
        score.no_speech += usize::from(provenance.no_speech);
    }
    for (language, score) in scores {
        println!(
            "{}",
            serde_json::json!({"language":language,"wer":score.word_errors as f64/score.words as f64,"cer":score.character_errors as f64/score.characters as f64,"counts":score})
        );
    }
    Ok(())
}
#[cfg(not(feature = "audio-stt"))]
fn main() {
    eprintln!("eval_stt requires audio-stt");
}
