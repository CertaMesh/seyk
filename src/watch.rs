//! Keep the cache fresh while files change.
//!
//! [`watch`] indexes once, then re-runs the same incremental index after each
//! burst of file changes. Re-indexing is cheap when little changed: unchanged
//! files are skipped by size and mtime, and only new segment texts reach the model.

use std::path::{Component, Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

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
    loop {
        // Block until something relevant changes, then wait for quiet.
        match rx.recv() {
            Ok(Ok(event)) if relevant(&event, &paths, discovery) => {}
            Ok(_) => continue,
            Err(_) => return Ok(()),
        }
        loop {
            match rx.recv_timeout(options.debounce) {
                Ok(_) => continue,
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
        run(&mut on_pass);
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
        // Judge only the part below the root: a root inside `~/.notes` is fine.
        let below = roots
            .iter()
            .find_map(|r| p.strip_prefix(r).ok())
            .unwrap_or(p);
        !ignored(below, discovery)
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
}
