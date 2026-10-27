//! Opt-in, content-free phase markers for external memory samplers.
//! Set SEYK_MEMORY_DIAGNOSTICS=1 before process startup. No source content,
//! token IDs, paths, errors or vectors are emitted.
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("SEYK_MEMORY_DIAGNOSTICS").is_some_and(|v| v == "1"))
}

fn marker(event: &'static str, state: &'static str, counts: &[(&'static str, usize)]) {
    if !enabled() {
        return;
    }
    static START: OnceLock<Instant> = OnceLock::new();
    let elapsed = START.get_or_init(Instant::now).elapsed().as_micros();
    let mut line = format!(
        "seyk_memory pid={} t_us={elapsed} event={event} state={state}",
        std::process::id()
    );
    for (key, value) in counts {
        use std::fmt::Write;
        // Writing to a String cannot fail.
        write!(line, " {key}={value}").expect("format diagnostic counts");
    }
    // Diagnostics must not change retrieval behavior on a closed stderr pipe.
    let _ = writeln!(std::io::stderr().lock(), "{line}");
}

/// Emit numeric workload metadata; names must be static program constants.
pub fn counts(event: &'static str, values: &[(&'static str, usize)]) {
    marker(event, "count", values);
}

/// A phase emits paired begin/end markers, including early error returns.
pub struct Phase(&'static str);
impl Phase {
    pub fn new(event: &'static str) -> Self {
        marker(event, "begin", &[]);
        Self(event)
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        marker(self.0, "end", &[]);
    }
}
