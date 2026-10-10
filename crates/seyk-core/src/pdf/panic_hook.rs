//! The hook is global, but suppression is confined to guarded parser work on
//! each thread. Concurrent unrelated panics retain the application's hook.
use std::cell::Cell;
use std::panic::PanicHookInfo;
use std::sync::{Arc, Mutex};

type Callback = Box<dyn Fn(&PanicHookInfo<'_>) + Send + Sync>;
struct Hook(Callback);
static STATE: Mutex<Option<(usize, Arc<Hook>)>> = Mutex::new(None);
thread_local! { static DEPTH: Cell<usize> = const { Cell::new(0) }; }

pub(super) struct Guard(std::marker::PhantomData<std::rc::Rc<()>>);
impl Guard {
    pub(super) fn new() -> Self {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((users, _)) = state.as_mut() {
            *users += 1;
        } else {
            let previous = Arc::new(Hook(std::panic::take_hook()));
            let delegate = previous.clone();
            std::panic::set_hook(Box::new(move |info| {
                if DEPTH.get() == 0 {
                    (delegate.0)(info);
                }
            }));
            *state = Some((1, previous));
        }
        DEPTH.set(DEPTH.get() + 1);
        Self(std::marker::PhantomData)
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        DEPTH.set(DEPTH.get() - 1);
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((users, _)) = state.as_mut() {
            *users -= 1;
            // Rust forbids replacing a hook while this thread is unwinding.
            // Leave the delegating hook installed with zero users in that rare
            // case; the next normal guard drop restores the original callback.
            if *users == 0 && !std::thread::panicking() {
                let (_, previous) = state.take().expect("active hook");
                // Drop the dispatcher before unwrapping its original callback;
                // restore that exact callback rather than accumulating wrappers.
                drop(std::panic::take_hook());
                let callback = match Arc::try_unwrap(previous) {
                    Ok(hook) => hook.0,
                    Err(hook) => Box::new(move |info: &PanicHookInfo<'_>| (hook.0)(info)),
                };
                std::panic::set_hook(callback);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scoped_hook_preserves_unrelated_and_concurrent_panics() {
        const CHILD: &str = "SEYK_TEST_SCOPED_PDF_HOOK";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "pdf::panic_hook::tests::scoped_hook_preserves_unrelated_and_concurrent_panics",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            return;
        }
        use std::sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        std::panic::set_hook(Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        let guard = super::Guard::new();
        assert!(std::panic::catch_unwind(|| panic!("private parser bytes")).is_err());
        let barrier = Arc::new(Barrier::new(2));
        let peer = barrier.clone();
        let worker = std::thread::spawn(move || {
            let _guard = super::Guard::new();
            peer.wait();
            assert!(std::panic::catch_unwind(|| panic!("another parser")).is_err());
            peer.wait();
        });
        barrier.wait();
        // A thread outside the parser scope must use the original hook even
        // while two parser guards are alive on other threads.
        assert!(std::thread::spawn(|| panic!("unrelated")).join().is_err());
        barrier.wait();
        worker.join().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(guard);
        assert!(std::panic::catch_unwind(|| panic!("restored")).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = super::Guard::new();
                panic!("guard also tolerates unwinding drop");
            })
            .is_err()
        );
        assert!(std::panic::catch_unwind(|| panic!("still delegated")).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        for _ in 0..100 {
            let _guard = super::Guard::new();
        }
        assert!(std::panic::catch_unwind(|| panic!("exact original restored")).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }
}
