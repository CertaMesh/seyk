//! One permanent dispatcher; only thread-local state changes during parsing.
//! A later external hook replaces ours. We never reinstall or remove it.
use std::cell::Cell;
use std::sync::OnceLock;

static INSTALLED: OnceLock<()> = OnceLock::new();
thread_local! {
    static SUPPRESSED: Cell<bool> = const { Cell::new(false) };
    static MESSAGE: Cell<Option<&'static str>> = const { Cell::new(None) };
}

// Never retain arbitrary upstream payloads: they can contain private source text.
fn diagnostic(payload: &(dyn std::any::Any + Send)) -> &'static str {
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
    match message {
        Some("unexpected entry in unicode map") => "unexpected entry in unicode map",
        Some("bad length of hexstring") => "bad length of hexstring",
        _ => "unrecognized parser panic",
    }
}

pub(super) struct Guard {
    previous: bool,
    message: Option<&'static str>,
    // TLS state must be restored on the same thread.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Guard {
    pub(super) fn new() -> Self {
        INSTALLED.get_or_init(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if SUPPRESSED.get() {
                    MESSAGE.set(Some(diagnostic(info.payload())));
                } else {
                    previous(info);
                }
            }));
        });
        Self {
            previous: SUPPRESSED.replace(true),
            message: MESSAGE.replace(None),
            _thread: std::marker::PhantomData,
        }
    }

    pub(super) fn message(&self) -> &'static str {
        MESSAGE.get().unwrap_or("unrecognized parser panic")
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        MESSAGE.set(self.message);
        SUPPRESSED.set(self.previous);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dispatcher_preserves_hook_ownership_and_unwinding() {
        const CHILD: &str = "SEYK_TEST_PDF_DISPATCHER";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "pdf::panic_hook::tests::dispatcher_preserves_hook_ownership_and_unwinding",
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
        // Installation is lazy and happens once, before racing parse transitions.
        drop(super::Guard::new());
        let barrier = Arc::new(Barrier::new(3));
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..1000 {
                        let guard = super::Guard::new();
                        assert!(
                            std::panic::catch_unwind(|| panic!("bad length of hexstring")).is_err()
                        );
                        assert_eq!(guard.message(), "bad length of hexstring");
                    }
                })
            })
            .collect();
        barrier.wait();
        for _ in 0..1000 {
            assert!(std::panic::catch_unwind(|| panic!("unrelated")).is_err());
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1000);
        {
            let guard = super::Guard::new();
            assert!(std::panic::catch_unwind(|| panic!("private source bytes")).is_err());
            assert_eq!(guard.message(), "unrecognized parser panic");
        }
        assert!(
            std::panic::catch_unwind(|| {
                let _guard = super::Guard::new();
                panic!("private bytes");
            })
            .is_err()
        );
        assert!(!super::SUPPRESSED.get());
        assert_eq!(super::MESSAGE.get(), None);
        assert!(std::panic::catch_unwind(|| panic!("after unwind")).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1001);
        let outer = super::Guard::new();
        {
            let _inner = super::Guard::new();
        }
        assert!(super::SUPPRESSED.get());
        // Install an external hook while a parse is active. Neither guard drop
        // nor later guard construction may overwrite it.
        let external = Arc::new(AtomicUsize::new(0));
        let counter = external.clone();
        std::panic::set_hook(Box::new(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        }));
        drop(outer);
        for _ in 0..10 {
            let _guard = super::Guard::new();
            assert!(std::panic::catch_unwind(|| panic!("external parser hook")).is_err());
        }
        assert!(std::panic::catch_unwind(|| panic!("external unrelated hook")).is_err());
        assert_eq!(external.load(Ordering::SeqCst), 11);
        assert_eq!(calls.load(Ordering::SeqCst), 1001);
    }
}
