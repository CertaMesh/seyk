//! Alone in its binary: it lowers the process descriptor limit.
#![cfg(unix)]

use seyk_core::DiscoveryOptions;
use seyk_core::discovery::discover;

fn open_fds() -> usize {
    std::fs::read_dir("/dev/fd").unwrap().count()
}

#[test]
fn deep_walks_keep_a_bounded_number_of_directories_open() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let mut deep = root.clone();
    // Deeper than the descriptor limit below, but short of PATH_MAX.
    for _ in 0..300 {
        deep = deep.join("d");
        std::fs::create_dir(&deep).unwrap();
        std::fs::write(deep.join("n.md"), "note").unwrap();
    }
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: writable rlimit output.
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
        0
    );
    let lowered = libc::rlimit {
        rlim_cur: 100,
        ..limit
    };
    let before = open_fds();
    assert!(before < 20, "{before} descriptors already open");
    // SAFETY: lowers only the soft limit, restored below.
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lowered) }, 0);
    let found = discover(&[root], &DiscoveryOptions::default());
    // SAFETY: restores the limit read above.
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
    assert_eq!(found.unwrap().len(), 300);
    assert_eq!(open_fds(), before);
}
