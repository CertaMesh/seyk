//! Explicit provisioning only; search never calls this module.
use super::{gemma, json_output};
use anyhow::Result;
use clap::ValueEnum;
use serde::Serialize;

const NEXT_STEP: &str = "seyk search \"...\" ./recordings";

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum Target {
    Audio,
}

#[derive(Serialize)]
struct ModelStatus {
    model: &'static str,
    present: bool,
    verification: &'static str,
    bytes: Option<u64>,
    repo: &'static str,
    revision: &'static str,
}

#[derive(Serialize)]
struct Report {
    action: &'static str,
    models: Vec<ModelStatus>,
    total_bytes: u64,
    download_bytes: u64,
    next_step: &'static str,
}

fn audio_status() -> ModelStatus {
    ModelStatus {
        model: "audio",
        present: gemma::verify_audio(&gemma::audio_directory()).is_ok(),
        verification: "sha256",
        bytes: Some(gemma::audio_assets().iter().map(|a| a.bytes).sum()),
        repo: gemma::ONNX_REPO,
        revision: gemma::ONNX_REVISION,
    }
}

fn stt_status() -> Result<ModelStatus> {
    use seyk::stt::manifest;
    Ok(ModelStatus {
        model: "stt",
        present: manifest::verify_directory(&manifest::default_directory()?).is_ok(),
        verification: "sha256",
        bytes: Some(manifest::assets().iter().map(|a| a.bytes).sum()),
        repo: manifest::REPO,
        revision: manifest::REVISION,
    })
}

fn next_step(models: &[ModelStatus]) -> &'static str {
    if models.iter().any(|m| m.model == "text" && !m.present) {
        "seyk model download"
    } else if models.iter().any(|m| !m.present) {
        "seyk setup audio"
    } else {
        NEXT_STEP
    }
}

fn report(action: &'static str, models: Vec<ModelStatus>) -> Report {
    Report {
        action,
        total_bytes: models.iter().filter_map(|m| m.bytes).sum(),
        download_bytes: models
            .iter()
            .filter(|m| !m.present)
            .filter_map(|m| m.bytes)
            .sum(),
        next_step: next_step(&models),
        models,
    }
}

fn print_plan(report: &Report) {
    eprintln!("Audio setup: pinned audio encoder and Whisper tiny STT");
    for model in &report.models {
        eprintln!(
            "  {}: {}@{} — {:.2} MiB ({})",
            model.model,
            model.repo,
            model.revision,
            model.bytes.unwrap_or(0) as f64 / 1048576.0,
            if model.present {
                "verified; skip"
            } else {
                "fetch and verify"
            },
        );
    }
    eprintln!(
        "Total: {:.2} MiB; to fetch: up to {:.2} MiB",
        report.total_bytes as f64 / 1048576.0,
        report.download_bytes as f64 / 1048576.0,
    );
}

// One provisioning boundary, injectable so tests never need real weights/network.
fn provision(
    models: &mut [ModelStatus],
    mut download: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    for model in models {
        if !model.present {
            download(model.model)?;
            model.present = true;
        }
    }
    Ok(())
}

pub(super) fn run(target: Option<Target>, dry_run: bool, json: bool) -> Result<()> {
    if target.is_some() && !dry_run {
        gemma::cleanup_audio_stages()?;
    }
    let models = vec![audio_status(), stt_status()?];
    let output = if target.is_some() {
        let mut plan = report("plan", models);
        print_plan(&plan);
        if !dry_run {
            provision(&mut plan.models, |model| {
                match model {
                    "audio" => {
                        gemma::download_audio()?;
                    }
                    "stt" => {
                        seyk::stt::download()?;
                    }
                    _ => unreachable!("only audio models are provisioned"),
                }
                Ok(())
            })?;
            plan.action = "complete";
            plan.next_step = next_step(&plan.models);
        }
        plan
    } else {
        let config = gemma::GemmaConfig::default();
        let mut models = models;
        models.insert(
            0,
            ModelStatus {
                model: "text",
                present: gemma::missing_model_files(&config)?.is_empty(),
                verification: "presence",
                bytes: None,
                repo: gemma::ONNX_REPO,
                revision: gemma::ONNX_REVISION,
            },
        );
        report("status", models)
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&json_output(&output))?);
    } else {
        for model in &output.models {
            println!(
                "{}: {} ({})",
                model.model,
                if model.present {
                    "ready"
                } else {
                    "missing or unverified"
                },
                model.verification
            );
        }
        if output
            .models
            .iter()
            .any(|m| m.model == "text" && !m.present)
        {
            println!("Text model: run seyk model download");
        }
        if target.is_none()
            && output
                .models
                .iter()
                .any(|m| m.model != "text" && !m.present)
        {
            println!("Audio models: run seyk setup audio");
        }
        println!("Next: {}", output.next_step);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_skips_verified_models_and_reruns_without_downloads() {
        let mut models = vec![
            ModelStatus {
                model: "audio",
                present: true,
                verification: "sha256",
                bytes: Some(10),
                repo: "test",
                revision: "pinned",
            },
            ModelStatus {
                model: "stt",
                present: false,
                verification: "sha256",
                bytes: Some(20),
                repo: "test",
                revision: "pinned",
            },
        ];
        let mut fetched = Vec::new();
        provision(&mut models, |model| {
            fetched.push(model.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(fetched, ["stt"]);
        provision(&mut models, |_| panic!("verified models must stay offline")).unwrap();
        assert!(models.iter().all(|m| m.present));
    }

    #[test]
    fn failed_download_is_not_reported_ready() {
        let mut models = vec![ModelStatus {
            model: "audio",
            present: false,
            verification: "sha256",
            bytes: Some(10),
            repo: "test",
            revision: "pinned",
        }];
        assert!(provision(&mut models, |_| anyhow::bail!("fake failure")).is_err());
        assert!(!models[0].present);
    }
}
