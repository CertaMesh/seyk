//! Standalone indexing telemetry. Counts come only from the engine's snapshots.
#[cfg(test)]
use crate::common;
use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use seyk::{IndexProgress, ProgressEvent};

#[derive(Clone, Copy, clap::ValueEnum)]
pub(super) enum Format {
    Jsonl,
}

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Discovering,
    LoadingModel,
    Indexing,
    Complete,
    Failed,
}

#[derive(serde::Serialize)]
struct Event<'a> {
    v: u8,
    r#type: &'static str,
    root: &'a str,
    phase: Phase,
    by_kind: &'a std::collections::BTreeMap<String, seyk::KindProgress>,
    done: usize,
    total: Option<usize>,
    eta_seconds: Option<u64>,
}

pub(super) struct JsonlProgress<W> {
    root: String,
    started: Instant,
    state: Mutex<State<W>>,
}

struct State<W> {
    writer: W,
    phase: Phase,
    snapshot: Option<IndexProgress>,
    last_emitted: Duration,
}

fn counts(progress: &IndexProgress) -> (usize, usize) {
    progress.by_kind.values().fold((0, 0), |(done, total), k| {
        let completed = k.indexed + k.skipped;
        (done + completed, total + completed + k.waiting)
    })
}

impl<W: Write> JsonlProgress<W> {
    pub(super) fn new(root: String, writer: W) -> Self {
        let progress = Self {
            root,
            started: Instant::now(),
            state: Mutex::new(State {
                writer,
                phase: Phase::Discovering,
                snapshot: None,
                last_emitted: Duration::ZERO,
            }),
        };
        progress.state.lock().unwrap().emit(&progress.root);
        progress
    }

    pub(super) fn loading_model(&self) {
        self.set_phase(Phase::LoadingModel);
    }

    pub(super) fn discovering(&self) {
        self.set_phase(Phase::Discovering);
    }

    fn set_phase(&self, phase: Phase) {
        let mut state = self.state.lock().unwrap();
        if state.phase != phase && !matches!(state.phase, Phase::Complete | Phase::Failed) {
            state.phase = phase;
            state.emit(&self.root);
        }
    }

    pub(super) fn event(&self, event: ProgressEvent) {
        self.event_at(event, self.started.elapsed());
    }

    fn event_at(&self, event: ProgressEvent, elapsed: Duration) {
        let ProgressEvent::IndexProgress { progress } = event else {
            return;
        };
        let mut state = self.state.lock().unwrap();
        if matches!(state.phase, Phase::Complete | Phase::Failed) {
            return;
        }
        // Concurrent callbacks can arrive after a newer snapshot. Never regress.
        if state
            .snapshot
            .as_ref()
            .is_some_and(|old| counts(old).0 > counts(&progress).0)
        {
            return;
        }
        state.snapshot = Some(progress);
        let phase_changed = state.phase != Phase::Indexing;
        state.phase = Phase::Indexing;
        if phase_changed || elapsed.saturating_sub(state.last_emitted) >= Duration::from_millis(250)
        {
            state.emit(&self.root);
            state.last_emitted = elapsed;
        }
    }

    pub(super) fn finish(&self, success: bool) {
        let mut state = self.state.lock().unwrap();
        if matches!(state.phase, Phase::Complete | Phase::Failed) {
            return;
        }
        state.phase = if success {
            Phase::Complete
        } else {
            Phase::Failed
        };
        if let Some(snapshot) = &mut state.snapshot {
            snapshot.eta_seconds = None;
        }
        state.emit(&self.root);
    }
}

impl<W: Write> State<W> {
    fn emit(&mut self, root: &str) {
        let empty = IndexProgress::default();
        let snapshot = self.snapshot.as_ref().unwrap_or(&empty);
        let (done, total) = counts(snapshot);
        let event = Event {
            v: 1,
            r#type: "progress",
            root,
            phase: self.phase,
            by_kind: &snapshot.by_kind,
            done,
            total: self.snapshot.as_ref().map(|_| total),
            eta_seconds: snapshot.eta_seconds,
        };
        // A closed telemetry pipe must not interrupt indexing or change stdout.
        if let Ok(mut line) = serde_json::to_vec(&event) {
            line.push(b'\n');
            let _ = self.writer.write_all(&line);
            let _ = self.writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn snapshot(indexed: usize, waiting: usize, skipped: usize) -> ProgressEvent {
        ProgressEvent::IndexProgress {
            progress: IndexProgress {
                by_kind: [(
                    "text".into(),
                    seyk::KindProgress {
                        indexed,
                        waiting,
                        skipped,
                    },
                )]
                .into(),
                eta_seconds: Some(2),
            },
        }
    }

    fn lines(progress: &JsonlProgress<Vec<u8>>) -> Vec<Value> {
        let state = progress.state.lock().unwrap();
        std::str::from_utf8(&state.writer)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn throttles_counts_but_keeps_phases_and_terminal_snapshot() {
        let progress = JsonlProgress::new("./notes\nquoted\"".into(), Vec::new());
        progress.event_at(snapshot(0, 5, 0), Duration::ZERO);
        for (ms, done) in [(1, 1), (249, 2), (250, 3), (251, 4), (500, 4)] {
            progress.event_at(snapshot(done, 5 - done, 0), Duration::from_millis(ms));
        }
        // Simulate a delayed older callback, and a completion inside the throttle window.
        progress.event_at(snapshot(1, 4, 0), Duration::from_millis(501));
        progress.event_at(snapshot(4, 0, 1), Duration::from_millis(502));
        progress.finish(true);
        progress.finish(true);
        progress.event(snapshot(0, 5, 0));
        let events = lines(&progress);
        assert_eq!(
            events
                .iter()
                .map(|e| e["phase"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "discovering",
                "indexing",
                "indexing",
                "indexing",
                "complete"
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|e| e["done"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [0, 0, 3, 4, 5]
        );
        assert!(events[0]["total"].is_null());
        for event in &events {
            assert_eq!(event["root"], "./notes\nquoted\"");
            assert_eq!(event["v"], 1);
            assert_eq!(event["type"], "progress");
            assert_eq!(event.as_object().unwrap().len(), 8);
        }
        for event in &events[1..] {
            assert_eq!(event["total"], 5);
        }
        assert_eq!(events[2]["eta_seconds"], 2);
        let final_event = events.last().unwrap();
        assert_eq!(final_event["total"], 5);
        assert_eq!(final_event["by_kind"]["text"]["skipped"], 1);
        assert!(final_event["eta_seconds"].is_null());
    }

    #[test]
    fn model_loading_and_discovery_changes_bypass_the_throttle() {
        let progress = JsonlProgress::new(".".into(), Vec::new());
        progress.loading_model();
        progress.discovering();
        progress.event_at(snapshot(0, 1, 0), Duration::ZERO);
        progress.event_at(snapshot(1, 0, 0), Duration::ZERO);
        progress.finish(true);
        let events = lines(&progress);
        assert_eq!(
            events
                .iter()
                .map(|e| e["phase"].as_str().unwrap())
                .collect::<Vec<_>>(),
            [
                "discovering",
                "loading_model",
                "discovering",
                "indexing",
                "complete"
            ]
        );
        assert!(events[..3].iter().all(|event| event["total"].is_null()));
    }

    #[test]
    #[should_panic]
    fn privacy_assertion_detects_escaped_windows_paths() {
        let private = r"C:\Users\private\notes";
        let serialized = serde_json::to_string(&serde_json::json!({"nested": [private]})).unwrap();
        let parsed = serde_json::from_str(&serialized).unwrap();
        common::assert_json_excludes(&parsed, private);
    }

    #[test]
    fn empty_pass_and_closed_stderr_do_not_interrupt_indexing() {
        let progress = JsonlProgress::new(".".into(), Vec::new());
        progress.event(ProgressEvent::IndexProgress {
            progress: IndexProgress::default(),
        });
        progress.finish(true);
        let events = lines(&progress);
        assert_eq!(events.len(), 3);
        assert_eq!(events[2]["phase"], "complete");
        assert_eq!(events[2]["done"], 0);
        assert_eq!(events[2]["total"], 0);

        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let closed = JsonlProgress::new(".".into(), Closed);
        closed.event(snapshot(0, 1, 0));
        closed.finish(false);
    }

    #[test]
    fn failure_is_terminal_and_does_not_claim_pending_resources_completed() {
        let progress = JsonlProgress::new(".".into(), Vec::new());
        progress.event(snapshot(1, 4, 0));
        progress.finish(false);
        let events = lines(&progress);
        assert_eq!(events.last().unwrap()["phase"], "failed");
        assert_eq!(events.last().unwrap()["done"], 1);
        assert!(events.last().unwrap()["eta_seconds"].is_null());
    }

    #[test]
    fn engine_progress_preserves_stdout_json_and_resource_privacy() {
        let temp = tempfile::tempdir().unwrap();
        for i in 0..5 {
            std::fs::write(
                temp.path().join(format!("private-name-{i}.txt")),
                "private source sentinel",
            )
            .unwrap();
        }
        std::fs::write(temp.path().join("private-image.png"), b"unsupported").unwrap();
        let index = |progress: Option<seyk::ProgressFn<'_>>| {
            let engine = seyk::SearchEngine::builder()
                .embedder(common::FakeEmbedder::new(seyk::Dimension::D256, "jsonl"))
                .batch_size(1)
                .build()
                .unwrap();
            let mut request = engine.index().paths([temp.path()]);
            if let Some(progress) = progress {
                request = request.progress(progress);
            }
            let mut report = request.execute().unwrap();
            // Timings naturally vary across runs; every other stdout byte must match.
            report.total_ms = 0;
            report.sync.embed_ms = 0;
            format!(
                "{}\n",
                serde_json::to_string_pretty(&crate::json_output(&report)).unwrap()
            )
        };
        let plain = index(None);
        let progress = JsonlProgress::new("./caller-root".into(), Vec::new());
        let callback = |event| progress.event(event);
        let observed = index(Some(&callback));
        progress.finish(true);
        assert_eq!(plain, observed);
        let events = lines(&progress);
        let mut last_done = 0;
        for event in &events {
            let done = event["done"].as_u64().unwrap();
            assert!(done >= last_done);
            last_done = done;
            for private in [
                "private-name",
                "private-image",
                "private source",
                temp.path().to_str().unwrap(),
            ] {
                common::assert_json_excludes(event, private);
            }
        }
        assert_eq!(last_done, 6);
        assert_eq!(events.last().unwrap()["by_kind"]["image"]["skipped"], 1);
    }
}
