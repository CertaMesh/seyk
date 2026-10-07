//! Keep the cache fresh while files change.
//!
//! [`watch`] indexes once, then re-runs the same incremental index after each
//! burst of file changes. Re-indexing is cheap when little changed: unchanged
//! files are skipped by size and mtime, and only new segment texts reach the model.

use std::path::{Component, Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use notify::{EventKind, RecursiveMode, Watcher};

use crate::discovery::DEFAULT_EXCLUDED_DIRS;
use crate::engine::{IndexReport, ProgressFn, SearchEngine, expand_tilde};
use crate::{DiscoveryOptions, Error, Result};

#[derive(Debug, Clone)]
pub struct WatchOptions {
    /// Wait this long after the last change before re-indexing, so a save burst
    /// or a `git checkout` triggers one pass instead of hundreds.
    pub debounce: Duration,
}

impl Default for WatchOptions {
    fn default() -> Self {
        Self {
            debounce: Duration::from_millis(1500),
        }
    }
}

/// Index `paths`, then keep indexing on change. Blocks until the watcher stops.
///
/// `on_pass` receives every pass's outcome, the first one included. Errors in a
/// pass are reported there and do not stop watching; failing to start does.
pub fn watch(
    engine: &SearchEngine,
    paths: &[PathBuf],
    discovery: &DiscoveryOptions,
    options: &WatchOptions,
    progress: Option<ProgressFn<'_>>,
    mut on_pass: impl FnMut(Result<IndexReport>),
) -> Result<()> {
    // Canonical, because the OS reports events with resolved paths (/private/var on macOS).
    let paths: Vec<PathBuf> = paths
        .iter()
        .map(|p| {
            let p = expand_tilde(p);
            p.canonicalize().unwrap_or(p)
        })
        .collect();
    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx).map_err(watch_error)?;
    for p in &paths {
        watcher
            .watch(p, RecursiveMode::Recursive)
            .map_err(watch_error)?;
    }

    let run = |on_pass: &mut dyn FnMut(Result<IndexReport>)| {
        let mut req = engine.index().paths(&paths).discovery(discovery.clone());
        if let Some(p) = progress {
            req = req.progress(p);
        }
        on_pass(req.execute());
    };

    // Watch before the first pass so edits made during it are not lost.
    run(&mut on_pass);
    debounce_loop(
        &rx,
        options.debounce,
        |event| matches!(event, Ok(e) if relevant(e, &paths, discovery)),
        || run(&mut on_pass),
    );
    Ok(())
}

/// Run `pass` once things have been quiet for `debounce` after a relevant event.
///
/// One deadline is the whole state: `None` while idle, `Some` once a relevant
/// event arrived. Only relevant events push it back, so build or VCS churn
/// cannot postpone a pending pass forever. Returns when the channel closes.
fn debounce_loop<E>(
    rx: &mpsc::Receiver<E>,
    debounce: Duration,
    relevant: impl Fn(&E) -> bool,
    mut pass: impl FnMut(),
) {
    // Far-future durations would overflow `Instant`; an hour is plenty.
    let debounce = debounce.min(Duration::from_secs(3600));
    let mut due: Option<Instant> = None;
    loop {
        let event = match due {
            None => match rx.recv() {
                Ok(e) => e,
                Err(_) => return,
            },
            // Check before receiving, so a backlog of ignored events cannot starve it.
            Some(d) if Instant::now() >= d => {
                due = None;
                pass();
                continue;
            }
            Some(d) => match rx.recv_timeout(d.saturating_duration_since(Instant::now())) {
                Ok(e) => e,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            },
        };
        if relevant(&event) {
            due = Some(Instant::now() + debounce);
        }
    }
}

fn watch_error(e: notify::Error) -> Error {
    Error::Config(format!("file watcher: {e}"))
}

/// Ignore reads and churn in folders discovery skips anyway (`target/` during a
/// build, `.git/` on every commit), so they do not trigger empty passes.
fn relevant(event: &notify::Event, roots: &[PathBuf], discovery: &DiscoveryOptions) -> bool {
    if matches!(event.kind, EventKind::Access(_)) {
        return false;
    }
    event.paths.iter().any(|p| {
        // Judge only the part below a root, so a root inside `~/.notes` is fine.
        // Roots can nest (`/repo` and `/repo/.notes`); any root admitting the
        // path is enough, whatever order the roots came in.
        let mut below = roots
            .iter()
            .filter_map(|r| p.strip_prefix(r).ok())
            .peekable();
        if below.peek().is_none() {
            return !ignored(p, discovery);
        }
        below.any(|b| !ignored(b, discovery))
    })
}

fn ignored(path: &Path, discovery: &DiscoveryOptions) -> bool {
    path.components().any(|c| match c {
        Component::Normal(name) => {
            let name = name.to_string_lossy();
            (discovery.default_excludes && DEFAULT_EXCLUDED_DIRS.contains(&name.as_ref()))
                || (!discovery.include_hidden && name.starts_with('.'))
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_vcs_churn_is_ignored() {
        let d = DiscoveryOptions::default();
        assert!(ignored(Path::new("/repo/target/debug/foo"), &d));
        assert!(ignored(Path::new("/repo/.git/index"), &d));
        assert!(ignored(Path::new("/repo/.DS_Store"), &d));
        assert!(!ignored(Path::new("/repo/src/main.rs"), &d));
        assert!(!ignored(Path::new("/notes/Chapter 1.md"), &d));
    }

    #[test]
    fn hidden_folders_above_the_root_do_not_matter() {
        let d = DiscoveryOptions::default();
        let event = notify::Event::new(EventKind::Any).add_path("/home/me/.notes/idea.md".into());
        assert!(relevant(&event, &["/home/me/.notes".into()], &d));
        assert!(!relevant(&event, &["/home/me".into()], &d));
    }

    #[test]
    fn nested_roots_admit_in_any_order() {
        let d = DiscoveryOptions::default();
        let event = notify::Event::new(EventKind::Any).add_path("/repo/.notes/idea.md".into());
        let parent: PathBuf = "/repo".into();
        let hidden: PathBuf = "/repo/.notes".into();
        assert!(relevant(&event, &[parent.clone(), hidden.clone()], &d));
        assert!(relevant(&event, &[hidden, parent], &d));
    }

    /// Events are (relevant, gap before sending). Returns how many passes ran.
    fn passes(events: Vec<(bool, Duration)>, debounce: Duration) -> usize {
        let (tx, rx) = mpsc::channel();
        let sender = std::thread::spawn(move || {
            for (relevant, gap) in events {
                std::thread::sleep(gap);
                tx.send(relevant).unwrap();
            }
            // Leave time for a pending pass before closing the channel.
            std::thread::sleep(Duration::from_millis(300));
        });
        let mut n = 0;
        debounce_loop(&rx, debounce, |&r| r, || n += 1);
        sender.join().unwrap();
        n
    }

    #[test]
    fn a_burst_of_relevant_changes_runs_one_pass() {
        let ms = Duration::from_millis;
        assert_eq!(passes(vec![(true, ms(0)); 20], ms(100)), 1);
    }

    #[test]
    fn ignored_churn_neither_triggers_nor_postpones_a_pass() {
        let ms = Duration::from_millis;
        assert_eq!(passes(vec![(false, ms(5)); 40], ms(50)), 0);
        // One real edit, then 600ms of ignored churn every 5ms: the pass must
        // still run once, 50ms after the edit, not after the churn stops.
        let mut events = vec![(true, ms(0))];
        events.extend(vec![(false, ms(5)); 120]);
        let started = Instant::now();
        let (tx, rx) = mpsc::channel();
        let sender = std::thread::spawn(move || {
            for (r, gap) in events {
                std::thread::sleep(gap);
                if tx.send(r).is_err() {
                    return;
                }
            }
        });
        let mut ran_at = None;
        debounce_loop(
            &rx,
            ms(50),
            |&r| r,
            || {
                ran_at.get_or_insert(started.elapsed());
            },
        );
        sender.join().unwrap();
        let ran_at = ran_at.expect("pass ran");
        assert!(
            ran_at < ms(400),
            "pass postponed by ignored churn: {ran_at:?}"
        );
    }
}
