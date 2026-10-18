use super::*;
use std::sync::atomic::Ordering;

const EXTERNAL_NAME: &str = "audio_encoder_quantized.onnx_data";

// Emit only this fixture's protobuf fields, allowing absolute external paths
// without an ONNX parser or Python dependency. Match the checked-in fixture below.
fn field(tag: u8, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let mut len = bytes.len();
    while len >= 128 {
        out.push((len as u8 & 127) | 128);
        len >>= 7;
    }
    out.push(len as u8);
    out.extend_from_slice(bytes);
    out
}

fn scalar_graph(location: &str, operation: &str) -> Vec<u8> {
    let node = [
        field(0x0a, b"x"),
        field(0x0a, b"weight"),
        field(0x12, b"y"),
        field(0x22, operation.as_bytes()),
    ]
    .concat();
    let mut weight = vec![0x08, 1, 0x10, 1]; // float[1]
    weight.extend(field(0x42, b"weight"));
    for (key, value) in [("location", location), ("offset", "0"), ("length", "4")] {
        weight.extend(field(
            0x6a,
            &[field(0x0a, key.as_bytes()), field(0x12, value.as_bytes())].concat(),
        ));
    }
    weight.extend([0x70, 1]); // EXTERNAL
    let mut graph = [
        field(0x0a, &node),
        field(0x12, b"external_scalar"),
        field(0x2a, &weight),
    ]
    .concat();
    for (tag, name) in [(0x5a, b"x"), (0x62, b"y")] {
        let value = [
            field(0x0a, name),
            vec![0x12, 10, 0x0a, 8, 0x08, 1, 0x12, 4, 0x0a, 2, 0x08, 1],
        ]
        .concat();
        graph.extend(field(tag, &value));
    }
    [
        vec![0x08, 8], // IR 8
        field(0x12, b"seyk regression fixture"),
        field(0x3a, &graph),
        vec![0x42, 2, 0x10, 18], // opset 18
    ]
    .concat()
}

fn files(dir: &Path) {
    std::fs::create_dir_all(dir.join("onnx")).unwrap();
    std::fs::write(dir.join(FILES[0]), scalar_graph(EXTERNAL_NAME, "Add")).unwrap();
    std::fs::write(dir.join(FILES[1]), 2.0_f32.to_le_bytes()).unwrap();
}

fn scalar(session: &mut Session) -> Result<f32> {
    let input = Tensor::from_array(([1usize], vec![0.0_f32])).map_err(ort_err)?;
    let out = session.run(ort::inputs!["x" => input]).map_err(ort_err)?;
    let (_, data) = out["y"].try_extract_tensor::<f32>().map_err(ort_err)?;
    Ok(data[0])
}

fn output(encoder: &AudioEncoder) -> f32 {
    encoder.with_session(Some(1), scalar).unwrap()
}

#[test]
fn tiny_fixture_builder_matches_checked_in_graph() {
    assert_eq!(
        scalar_graph("weight.bin", "Add"),
        include_bytes!("../../tests/fixtures/ort/external_scalar.onnx").as_slice()
    );
}

#[test]
fn captured_bytes_survive_rename_replacement() {
    for name in FILES {
        let dir = tempfile::tempdir().unwrap();
        files(dir.path());
        let encoder = AudioEncoder::new(dir.path());
        let identity = encoder.identity().unwrap().to_owned();
        let replacement = dir.path().join("replacement");
        let bytes = if name == FILES[0] {
            scalar_graph(EXTERNAL_NAME, "Sub")
        } else {
            9.0_f32.to_le_bytes().to_vec()
        };
        assert_eq!(
            bytes.len() as u64,
            std::fs::metadata(dir.path().join(name)).unwrap().len()
        );
        std::fs::write(&replacement, bytes).unwrap();
        // Windows rename cannot replace an existing target. Move the old name
        // aside first; no writer or test depends on filesystem scheduling.
        std::fs::rename(dir.path().join(name), dir.path().join("old")).unwrap();
        std::fs::rename(replacement, dir.path().join(name)).unwrap();
        assert_eq!(output(&encoder), 2.0);
        assert_eq!(encoder.identity().unwrap(), identity);
        let fresh = AudioEncoder::new(dir.path());
        assert_ne!(fresh.identity().unwrap(), identity);
        assert_eq!(output(&fresh), if name == FILES[0] { -2.0 } else { 9.0 });
        assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn captured_bytes_survive_in_place_and_hard_link_mutation() {
    for name in FILES {
        for through_link in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            files(dir.path());
            let encoder = AudioEncoder::new(dir.path());
            let identity = encoder.identity().unwrap().to_owned();
            let original = dir.path().join(name);
            let path = if through_link {
                let alias = dir.path().join("alias");
                std::fs::hard_link(&original, &alias).unwrap();
                alias
            } else {
                original
            };
            let metadata = std::fs::metadata(&path).unwrap();
            let bytes = if name == FILES[0] {
                scalar_graph(EXTERNAL_NAME, "Sub")
            } else {
                9.0_f32.to_le_bytes().to_vec()
            };
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.write_all(&bytes).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
                .unwrap();
            assert_eq!(file.metadata().unwrap().len(), metadata.len());
            assert_eq!(
                file.metadata().unwrap().modified().unwrap(),
                metadata.modified().unwrap()
            );
            drop(file);
            assert_eq!(output(&encoder), 2.0);
            assert_eq!(encoder.identity().unwrap(), identity);
            let fresh = AudioEncoder::new(dir.path());
            assert_ne!(fresh.identity().unwrap(), identity);
            assert_eq!(output(&fresh), if name == FILES[0] { -2.0 } else { 9.0 });
        }
    }
}

#[test]
#[cfg(unix)]
fn captured_regular_symlink_targets_survive_retargeting() {
    for name in FILES {
        let dir = tempfile::tempdir().unwrap();
        files(dir.path());
        let path = dir.path().join(name);
        let original = dir.path().join("original");
        std::fs::rename(&path, &original).unwrap();
        std::os::unix::fs::symlink(&original, &path).unwrap();
        let encoder = AudioEncoder::new(dir.path());
        let identity = encoder.identity().unwrap().to_owned();
        let other = dir.path().join("other");
        let bytes = if name == FILES[0] {
            scalar_graph(EXTERNAL_NAME, "Sub")
        } else {
            9.0_f32.to_le_bytes().to_vec()
        };
        std::fs::write(&other, bytes).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&other, &path).unwrap();
        assert_eq!(output(&encoder), 2.0);
        let fresh = AudioEncoder::new(dir.path());
        assert_ne!(fresh.identity().unwrap(), identity);
        assert_eq!(output(&fresh), if name == FILES[0] { -2.0 } else { 9.0 });
    }
}

#[test]
fn deleted_paths_and_dropped_builder_do_not_affect_inference() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let encoder = AudioEncoder::new(dir.path());
    encoder.identity().unwrap();
    for name in FILES {
        std::fs::remove_file(dir.path().join(name)).unwrap();
    }
    let mut session = encoder.snapshot().unwrap().session(Some(1)).unwrap();
    // The production helper has already dropped the builder and its Cow.
    drop(encoder);
    for _ in 0..2 {
        assert_eq!(scalar(&mut session).unwrap(), 2.0);
    }
}

#[test]
fn concurrent_identity_and_session_requests_share_one_capture_and_session() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let encoder = AudioEncoder::new(dir.path());
    let start = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let encoder = &encoder;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    if worker % 2 == 0 {
                        assert!(!encoder.identity().unwrap().is_empty());
                    } else {
                        assert_eq!(output(encoder), 2.0);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 1);
    assert_eq!(encoder.session_loads.load(Ordering::Relaxed), 1);
}

#[test]
fn observed_growth_truncation_or_mtime_change_publishes_nothing_and_retries() {
    for name in FILES {
        for mutation in 0..3 {
            let dir = tempfile::tempdir().unwrap();
            files(dir.path());
            let encoder = AudioEncoder::new(dir.path());
            let result = encoder.prepare_with(|path, point| {
                if path == dir.path().join(name) {
                    if mutation < 2 && point == CapturePoint::Opened {
                        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
                        let len = file.metadata().unwrap().len();
                        file.set_len(if mutation == 0 { len + 4096 } else { len - 1 })
                            .unwrap();
                    } else if mutation == 2 && point == CapturePoint::Read {
                        std::fs::OpenOptions::new()
                            .write(true)
                            .open(path)
                            .unwrap()
                            .set_modified(std::time::SystemTime::UNIX_EPOCH)
                            .unwrap();
                    }
                }
                Ok(())
            });
            assert!(
                matches!(result, Err(Error::Embedding(message)) if message.contains("asset changed"))
            );
            assert!(encoder.snapshot.get().is_none());
            assert!(encoder.session.lock().unwrap().is_none());
            files(dir.path());
            assert_eq!(output(&encoder), 2.0);
            assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 2);
        }
    }
}

#[test]
fn injected_capture_failure_between_assets_is_retryable() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let encoder = AudioEncoder::new(dir.path());
    assert!(
        encoder
            .prepare_with(|path, point| {
                if path == dir.path().join(FILES[1]) && point == CapturePoint::Opened {
                    return Err(io_error(
                        path,
                        std::io::Error::other("injected read failure"),
                    ));
                }
                Ok(())
            })
            .is_err()
    );
    assert!(encoder.snapshot.get().is_none());
    assert_eq!(output(&encoder), 2.0);
    assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 2);
}

#[test]
fn failed_session_creation_keeps_the_pinned_snapshot_for_retry() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    std::fs::write(dir.path().join(FILES[0]), b"not an ONNX graph").unwrap();
    let encoder = AudioEncoder::new(dir.path());
    let identity = encoder.identity().unwrap().to_owned();
    files(dir.path());
    for _ in 0..2 {
        assert!(encoder.with_session(Some(1), scalar).is_err());
        assert!(encoder.session.lock().unwrap().is_none());
        assert_eq!(encoder.identity().unwrap(), identity);
    }
    assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 1);
    assert_eq!(output(&AudioEncoder::new(dir.path())), 2.0);
}

#[test]
fn unregistered_relative_and_absolute_external_references_fail_closed() {
    let cwd = std::env::current_dir().unwrap();
    let dir = tempfile::tempdir_in(&cwd).unwrap();
    files(dir.path());
    let disk_weights = dir.path().join("unregistered.bin");
    let relative = disk_weights.strip_prefix(&cwd).unwrap();
    assert!(relative.is_relative());
    assert!(disk_weights.is_absolute());
    std::fs::write(&disk_weights, 9.0_f32.to_le_bytes()).unwrap();
    assert_eq!(cwd.join(relative), disk_weights);
    for path in [relative, disk_weights.as_path()] {
        assert_eq!(std::fs::read(path).unwrap(), 9.0_f32.to_le_bytes());
    }
    for location in [relative.to_str().unwrap(), disk_weights.to_str().unwrap()] {
        std::fs::write(dir.path().join(FILES[0]), scalar_graph(location, "Add")).unwrap();
        let encoder = AudioEncoder::new(dir.path());
        let identity = encoder.identity().unwrap().to_owned();
        for _ in 0..2 {
            assert!(
                encoder.with_session(Some(1), scalar).is_err(),
                "memory session must reject unregistered external reference {location}"
            );
            assert!(encoder.session.lock().unwrap().is_none());
            assert_eq!(encoder.identity().unwrap(), identity);
        }
    }
}

#[test]
fn missing_or_truncated_registered_external_bytes_fail_without_path_fallback() {
    for bytes in [Vec::new(), vec![0; 3]] {
        let dir = tempfile::tempdir().unwrap();
        files(dir.path());
        std::fs::write(dir.path().join(FILES[1]), bytes).unwrap();
        let encoder = AudioEncoder::new(dir.path());
        let identity = encoder.identity().unwrap().to_owned();
        // Even repairing the path must not repair this pinned capture.
        files(dir.path());
        assert!(encoder.with_session(Some(1), scalar).is_err());
        assert!(encoder.session.lock().unwrap().is_none());
        assert_eq!(encoder.identity().unwrap(), identity);
        assert_eq!(output(&AudioEncoder::new(dir.path())), 2.0);
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "measures RSS with pinned local text/audio weights; never downloads"]
fn snapshot_rss_stages() {
    fn sample(stage: &str, model: Option<&EmbeddingGemma2>) {
        let pid = libc::pid_t::try_from(std::process::id()).unwrap();
        let size = libc::c_int::try_from(std::mem::size_of::<libc::proc_taskinfo>()).unwrap();
        let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::uninit();
        // SAFETY: info points to a writable, aligned buffer of exactly size bytes.
        let returned = unsafe {
            libc::proc_pidinfo(pid, libc::PROC_PIDTASKINFO, 0, info.as_mut_ptr().cast(), size)
        };
        if returned != size {
            let error = std::io::Error::last_os_error();
            panic!("proc_pidinfo returned {returned}/{size} bytes: {error}");
        }
        // SAFETY: a full PROC_PIDTASKINFO result initializes every proc_taskinfo field.
        let resident_bytes = unsafe { info.assume_init() }.pti_resident_size;
        assert!(resident_bytes > 0);
        let rss_kib = resident_bytes / 1024;
        let (owned_bytes, session_loads) = model
            .and_then(|model| model.audio.as_ref())
            .map(|audio| {
                let bytes = audio
                    .snapshot
                    .get()
                    .map(|snapshot| snapshot.graph.len() + snapshot.external.len())
                    .unwrap_or(0);
                (bytes, audio.session_loads.load(Ordering::Relaxed))
            })
            .unwrap_or((0, 0));
        println!(
            "AUDIO_SNAPSHOT_RSS stage={stage} rss_kib={rss_kib} owned_bytes={owned_bytes} session_loads={session_loads}"
        );
    }

    sample("start", None);
    let model = EmbeddingGemma2::builder()
        .vision(false)
        .audio(true)
        .threads(1)
        .build()
        .unwrap();
    model.embed_query("synthetic text-only query").unwrap();
    let audio = model.audio.as_ref().unwrap();
    assert!(audio.snapshot.get().is_none());
    assert_eq!(audio.bytes_read.load(Ordering::Relaxed), 0);
    sample("text-only", Some(&model));
    model.input_identity(Modality::Audio).unwrap();
    assert_eq!(audio.session_loads.load(Ordering::Relaxed), 0);
    sample("prepared-no-session", Some(&model));
    let signal: Vec<f32> = include_bytes!("../../tests/fixtures/audio/audio_signal.f32")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let first = model.embed_audio(&signal, 16000).unwrap();
    sample("first-inference", Some(&model));
    assert_eq!(model.embed_audio(&signal, 16000).unwrap(), first);
    sample("repeat-inference", Some(&model));
    assert_eq!(audio.hash_passes.load(Ordering::Relaxed), 1);
    assert_eq!(audio.session_loads.load(Ordering::Relaxed), 1);
    drop(model);
    // Allocators may retain freed pages. Only this current-RSS sample, rather
    // than the process high-water mark, describes the state after model drop.
    sample("model-dropped", None);
}
