//! Resource-level progress for one indexing pass. No persistent cache state.
use crate::discovery::ResourceKind;
use crate::{ProgressEvent, ProgressFn, ProvidedResource, ResourceId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Counts of discovered resources in the current (or last completed) pass.
/// A mixed PDF is one `pdf` resource, regardless of its page locator kinds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindProgress {
    pub indexed: usize,
    pub waiting: usize,
    pub skipped: usize,
}

/// Snapshot of a pass, not a durable inventory. Counts reset on rediscovery.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexProgress {
    pub by_kind: BTreeMap<String, KindProgress>,
    /// Remaining resource count / observed newly indexed resources per second.
    /// Null until two new resources complete and at least 1 ms has elapsed;
    /// cached reuse and skips never train the estimate. Null when finished.
    pub eta_seconds: Option<u64>,
}

pub(crate) struct PassProgress<'a> {
    state: Mutex<PassState>,
    callback: Option<ProgressFn<'a>>,
    started: Instant,
}
struct PassState {
    pending: HashMap<ResourceId, &'static str>,
    snapshot: IndexProgress,
    updated: usize,
}
impl<'a> PassProgress<'a> {
    pub(crate) fn new(resources: &[ProvidedResource], callback: Option<ProgressFn<'a>>) -> Self {
        let mut state = PassState {
            pending: HashMap::new(),
            snapshot: IndexProgress::default(),
            updated: 0,
        };
        for r in resources {
            let kind = match r.kind {
                ResourceKind::Text(_) => "text",
                ResourceKind::Pdf => "pdf",
                ResourceKind::Image => "image",
                ResourceKind::Audio => "audio",
            };
            if let std::collections::hash_map::Entry::Vacant(entry) =
                state.pending.entry(r.id.clone())
            {
                entry.insert(kind);
                state
                    .snapshot
                    .by_kind
                    .entry(kind.into())
                    .or_default()
                    .waiting += 1;
            }
        }
        let tracker = Self {
            state: Mutex::new(state),
            callback,
            started: Instant::now(),
        };
        tracker.emit();
        tracker
    }
    fn emit(&self) {
        if let Some(callback) = self.callback {
            let progress = self.snapshot();
            callback(ProgressEvent::IndexProgress { progress });
        }
    }
    pub(crate) fn complete(&self, id: &ResourceId, indexed: bool, updated: bool) {
        {
            let mut state = self.state.lock().unwrap();
            let Some(kind) = state.pending.remove(id) else {
                return;
            };
            let counts = state.snapshot.by_kind.get_mut(kind).unwrap();
            counts.waiting -= 1;
            if indexed {
                counts.indexed += 1;
            } else {
                counts.skipped += 1;
            }
            state.updated += usize::from(updated);
        }
        self.emit();
    }
    pub(crate) fn snapshot(&self) -> IndexProgress {
        self.snapshot_after(self.started.elapsed())
    }
    fn snapshot_after(&self, elapsed: Duration) -> IndexProgress {
        let state = self.state.lock().unwrap();
        let mut result = state.snapshot.clone();
        result.eta_seconds = estimate(state.pending.len(), state.updated, elapsed);
        result
    }
}
fn estimate(waiting: usize, updated: usize, elapsed: Duration) -> Option<u64> {
    if waiting == 0 || updated < 2 || elapsed < Duration::from_millis(1) {
        return None;
    }
    Some((waiting as f64 * elapsed.as_secs_f64() / updated as f64).ceil() as u64)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eta_requires_observations_and_resets_at_completion() {
        assert_eq!(estimate(10, 0, Duration::from_secs(10)), None);
        assert_eq!(estimate(10, 1, Duration::from_secs(10)), None);
        assert_eq!(estimate(10, 2, Duration::ZERO), None);
        assert_eq!(estimate(10, 2, Duration::from_secs(4)), Some(20));
        assert_eq!(estimate(6, 4, Duration::from_secs(6)), Some(9));
        assert_eq!(estimate(0, 4, Duration::from_secs(6)), None);
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    #[test]
    fn eta_tracks_real_completion_transitions_with_a_controlled_clock() {
        let mut resources: Vec<_> = (0..5)
            .map(|i| ProvidedResource {
                id: format!("mem://test/{i}").parse().unwrap(),
                kind: ResourceKind::Text(crate::segment::TextFormat::Plain),
                size_bytes: 1,
                modified_ns: None,
                version: Default::default(),
                title: None,
            })
            .collect();
        // A provider repeating an ID must not double-count or replace its kind.
        let mut duplicate = resources[0].clone();
        duplicate.kind = ResourceKind::Image;
        resources.push(duplicate);
        let tracker = PassProgress::new(&resources, None);
        assert_eq!(tracker.snapshot().by_kind.len(), 1);
        let elapsed = Duration::from_secs(4);
        assert_eq!(tracker.snapshot_after(elapsed).eta_seconds, None);
        tracker.complete(&resources[0].id, true, false); // reuse is not a sample
        tracker.complete(&resources[1].id, false, false); // nor is a skip
        tracker.complete(&resources[2].id, true, true);
        assert_eq!(tracker.snapshot_after(elapsed).eta_seconds, None);
        tracker.complete(&resources[3].id, true, true);
        assert_eq!(tracker.snapshot_after(elapsed).eta_seconds, Some(2));
        tracker.complete(&resources[4].id, true, true);
        assert_eq!(tracker.snapshot_after(elapsed).eta_seconds, None);
        let counts = &tracker.snapshot_after(elapsed).by_kind["text"];
        assert_eq!((counts.indexed, counts.waiting, counts.skipped), (4, 0, 1));
    }
}
