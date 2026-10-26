use super::*;
use std::sync::Barrier;

// ONNX IR 8 / opset 18: y = x + weight, each a one-element f32 tensor.
// The initializer reads four bytes from weight.bin beside the graph. Each
// directory has a different value so a valid but wrong path also fails.
const EXTERNAL_SCALAR: &[u8] = include_bytes!("../tests/fixtures/ort/external_scalar.onnx");

#[test]
fn concurrent_external_data_sessions_keep_their_model_directories() {
    const WORKERS: usize = 8;
    let root = tempfile::tempdir().unwrap();
    let paths: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let dir = root
                .path()
                .join(format!("model-{worker}-{}", "x".repeat(worker * 17)));
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join("model.onnx"), EXTERNAL_SCALAR).unwrap();
            std::fs::write(dir.join("weight.bin"), ((worker + 1) as f32).to_le_bytes()).unwrap();
            dir.join("model.onnx")
        })
        .collect();
    let start = Barrier::new(WORKERS);
    std::thread::scope(|scope| {
        let handles: Vec<_> = paths
            .iter()
            .enumerate()
            .map(|(worker, path)| {
                let start = &start;
                scope.spawn(move || -> Result<()> {
                    start.wait();
                    for _ in 0..16 {
                        let mut model = session(path, Some(1))?;
                        let input =
                            Tensor::from_array(([1usize], vec![0.0_f32])).map_err(ort_err)?;
                        let outputs = model.run(ort::inputs!["x" => input]).map_err(ort_err)?;
                        let (_, actual) =
                            outputs["y"].try_extract_tensor::<f32>().map_err(ort_err)?;
                        assert_eq!(
                            actual,
                            &[(worker + 1) as f32],
                            "initializer came from another model directory"
                        );
                    }
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
    });
}

#[test]
fn allocator_policies_load_external_weights_and_run_without_numerical_changes() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model.onnx");
    std::fs::write(&path, EXTERNAL_SCALAR).unwrap();
    std::fs::write(root.path().join("weight.bin"), 7.0_f32.to_le_bytes()).unwrap();
    for policy in [
        CpuAllocator::Default,
        CpuAllocator::NoArena,
        CpuAllocator::ArenaShrink,
    ] {
        let mut model = session_with_allocator(&path, Some(1), policy).unwrap();
        let options = text_run_options(policy).unwrap();
        for _ in 0..3 {
            let input = Tensor::from_array(([1usize], vec![2.0_f32])).unwrap();
            let outputs = match options.as_ref() {
                Some(options) => model.run_with_options(ort::inputs!["x" => input], options),
                None => model.run(ort::inputs!["x" => input]),
            }
            .unwrap();
            assert_eq!(outputs["y"].try_extract_tensor::<f32>().unwrap().1, &[9.0]);
        }
    }
}
