#![cfg(all(target_os = "macos", feature = "audio"))]

use seyk_embeddinggemma::{
    Embedder, Embedding, EmbeddingGemma2, GemmaConfig, Modality, ONNX_REPO, ONNX_REVISION,
    Precision, audio, locate,
};
use std::io::Read;

fn current_rss_kib() -> u64 {
    let pid = libc::pid_t::try_from(std::process::id()).unwrap();
    let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_taskinfo>()).unwrap();
    let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::uninit();
    // SAFETY: info is an aligned writable buffer of exactly size bytes.
    let returned = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTASKINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    if returned != size {
        let error = std::io::Error::last_os_error();
        panic!("proc_pidinfo returned {returned}/{size} bytes: {error}");
    }
    // SAFETY: only a full PROC_PIDTASKINFO result permits reading initialized fields.
    let resident = unsafe { info.assume_init() }.pti_resident_size;
    assert!(resident > 0);
    resident / 1024
}

fn sample(stage: &str) {
    println!(
        "MATCHED_MEMORY_STAGE stage={stage} rss_kib={}",
        current_rss_kib()
    );
}

fn independent_identity(config: &GemmaConfig) -> String {
    let dir = locate(config).unwrap();
    let mut hash = blake3::Hasher::new();
    hash.update(audio::FEATURE_POLICY.as_bytes());
    for name in audio::FILES {
        hash.update(name.as_bytes());
        let mut file = std::fs::File::open(dir.join(name)).unwrap();
        let mut buffer = [0; 65536];
        loop {
            let count = file.read(&mut buffer).unwrap();
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
    }
    hash.finalize().to_hex().to_string()
}

fn check_output(output: &Embedding) {
    assert_eq!(output.as_slice().len(), 768);
    assert!(output.as_slice().iter().all(|value| value.is_finite()));
    assert!(output.normalized());
    std::hint::black_box(output.as_slice().iter().map(|v| f64::from(*v)).sum::<f64>());
}

fn cosine(output: &Embedding, reference: &[f32]) -> f64 {
    assert_eq!(reference.len(), 768);
    assert!(reference.iter().all(|v| v.is_finite()));
    let dot: f64 = output
        .as_slice()
        .iter()
        .zip(reference)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let a = output
        .as_slice()
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    let b = reference
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(a > 0.0 && b > 0.0);
    let result = dot / (a * b);
    assert!(result.is_finite() && result >= 0.99);
    result
}

fn bits(output: &Embedding) -> Vec<u32> {
    output.as_slice().iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "provisional matched memory evidence; requires explicit pinned offline model grant"]
fn matched_memory_six_stages() {
    let bytes = include_bytes!("fixtures/audio/audio_signal.f32");
    assert_eq!(bytes.len(), 16000 * 4);
    let signal: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert!(signal.iter().all(|v| v.is_finite()));
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/audio/reference.json")).unwrap();
    let expected: Vec<f32> = serde_json::from_value(reference["embedding"].clone()).unwrap();
    let mut config = GemmaConfig::default();
    config.vision = false;
    config.audio = true;
    config.threads = Some(1);
    let expected_identity = independent_identity(&config);
    sample("start");
    let model = EmbeddingGemma2::builder()
        .dimension(768)
        .precision(Precision::Q8)
        .vision(false)
        .audio(true)
        .threads(1)
        .sessions(1)
        .batch_size(16)
        .max_tokens(2048)
        .build()
        .unwrap();
    assert_eq!(model.config().threads, Some(1));
    assert_eq!(model.config().sessions, 1);
    assert_eq!(model.config().batch_size, 16);
    assert_eq!(model.config().max_tokens, 2048);
    assert_eq!(model.config().precision, Precision::Q8);
    assert!(!model.config().vision && model.config().audio);
    let model_identity = model.identity().clone();
    assert_eq!(
        model_identity.revision,
        format!("{ONNX_REPO}@{ONNX_REVISION}")
    );
    assert_eq!(model_identity.dimension.get(), 768);
    let text = model.embed_query("synthetic text-only query").unwrap();
    check_output(&text);
    assert_eq!(model.input_identity(Modality::Text).unwrap(), "");
    sample("text-only");
    let identity = model.input_identity(Modality::Audio).unwrap().to_owned();
    assert_eq!(identity, expected_identity);
    assert_eq!(model.input_identity(Modality::Audio).unwrap(), identity);
    sample("prepared-no-session");
    let first = model.embed_audio(&signal, 16000).unwrap();
    check_output(&first);
    let first_cosine = cosine(&first, &expected);
    sample("first-inference");
    let repeat = model.embed_audio(&signal, 16000).unwrap();
    check_output(&repeat);
    assert_eq!(first, repeat);
    let repeat_cosine = cosine(&repeat, &expected);
    assert_eq!(model.input_identity(Modality::Audio).unwrap(), identity);
    sample("repeat-inference");
    drop(model);
    sample("model-dropped");
    // Match the live small buffers on both sides through the final sample.
    std::hint::black_box((
        &signal,
        &reference,
        &expected,
        &text,
        &first,
        &repeat,
        &identity,
        &expected_identity,
    ));
    println!(
        "MATCHED_MEMORY_RESULT {}",
        serde_json::json!({
            "schema": 1, "workload": "one-second-q8-768-threads1-sessions1",
            "errors": 0, "text_calls": 1, "audio_calls": 2,
            "model_identity": model_identity, "audio_identity": identity,
            "independent_audio_identity": expected_identity,
            "first_cosine": first_cosine, "repeat_cosine": repeat_cosine,
            "text_bits": bits(&text), "first_bits": bits(&first), "repeat_bits": bits(&repeat),
            "config": {"dimension":768,"precision":"q8","vision":false,"audio":true,
                "threads":1,"sessions":1,"batch_size":16,"max_tokens":2048}
        })
    );
}
