use super::*;
use crate::local_content::LocalContentProvider;
use crate::{DiscoveryOptions, ResourceId, ResourceProvider, ResourceScope};
use std::cell::{Cell, RefCell};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;

struct Hook {
    parent: (u64, u64),
    name: OsString,
    run: Box<dyn FnOnce()>,
}

thread_local! {
    static HOOKS: RefCell<Vec<Hook>> = const { RefCell::new(Vec::new()) };
}

// Exact parent inode + basename, one-shot, and thread-local: unrelated opens
// and parallel fixtures cannot consume a callback. No hook exists in release.
pub(super) fn opening(directory: &File, name: &OsStr) {
    let parent = identity(&descriptor_stat(directory).unwrap());
    let hook = HOOKS.with_borrow_mut(|hooks| {
        hooks
            .iter()
            .position(|h| h.parent == parent && h.name == name)
            .map(|i| hooks.remove(i).run)
    });
    if let Some(run) = hook {
        run();
    }
}

fn before_open(path: &Path, run: impl FnOnce() + 'static) -> Rc<Cell<bool>> {
    let parent = File::open(path.parent().unwrap()).unwrap();
    let fired = Rc::new(Cell::new(false));
    let observed = fired.clone();
    HOOKS.with_borrow_mut(|hooks| {
        hooks.push(Hook {
            parent: identity(&descriptor_stat(&parent).unwrap()),
            name: path.file_name().unwrap().to_owned(),
            run: Box::new(move || {
                observed.set(true);
                run();
            }),
        });
    });
    fired
}

fn fixture() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::Builder::new()
        .prefix("local-content-")
        .tempdir()
        .unwrap();
    let base = temp.path().canonicalize().unwrap();
    (temp, base)
}

fn scope(root: &Path) -> ResourceScope {
    ResourceScope {
        roots: vec![ResourceId::from_path(root).unwrap()],
        discovery: DiscoveryOptions::default(),
    }
}

fn assert_content(provider: &LocalContentProvider, root: &Path) {
    let id = ResourceId::from_path(root.join("note.rs")).unwrap();
    assert_eq!(provider.read(&id).unwrap(), b"public replacement sentinel");
    let mut bytes = Vec::new();
    provider.open(&id).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, b"public replacement sentinel");
    assert!(provider.version(&id).is_ok());
    let listed = provider.list(&scope(root)).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, id);
}

fn assert_replaced_root_refused(provider: &LocalContentProvider, root: &Path) {
    fs::rename(root, root.with_file_name("retired-root")).unwrap();
    fs::create_dir(root).unwrap();
    fs::write(root.join("note.rs"), b"public unwanted replacement").unwrap();
    let id = ResourceId::from_path(root.join("note.rs")).unwrap();
    assert!(provider.read(&id).is_err());
    assert!(provider.open(&id).is_err());
    assert!(provider.version(&id).is_err());
    assert!(provider.list(&scope(root)).is_err());
}

#[test]
fn local_content_adopts_directory_open_identity_at_root_and_ancestor() {
    for ancestor in [false, true] {
        let (_temp, base) = fixture();
        let component = base.join("collection");
        let root = if ancestor {
            component.join("selected")
        } else {
            component.clone()
        };
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("old.rs"), b"public old sentinel").unwrap();
        let exchanged = component.clone();
        let selected = root.clone();
        let fired = before_open(&component, move || {
            fs::rename(&exchanged, exchanged.with_file_name("retired")).unwrap();
            fs::create_dir_all(&selected).unwrap();
            fs::write(selected.join("note.rs"), b"public replacement sentinel").unwrap();
        });
        let result = LocalContentProvider::new(std::slice::from_ref(&root));
        assert!(fired.get());
        let provider = result.unwrap();
        assert_content(&provider, &root);
        assert_replaced_root_refused(&provider, &root);
    }
}

#[test]
fn local_content_directory_open_refuses_swapped_links_and_special_files() {
    for kind in ["symlink", "fifo", "socket", "file"] {
        let (_temp, base) = fixture();
        let root = base.join("collection");
        fs::create_dir(&root).unwrap();
        let outside = base.join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("note.rs"), b"public outside sentinel").unwrap();
        let exchanged = root.clone();
        let fired = before_open(&root, move || {
            fs::rename(&exchanged, exchanged.with_file_name("retired")).unwrap();
            match kind {
                "symlink" => std::os::unix::fs::symlink(outside, exchanged).unwrap(),
                "fifo" => {
                    let path = c_name(exchanged.as_os_str()).unwrap();
                    // SAFETY: valid NUL-terminated fixture pathname.
                    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
                }
                "socket" => {
                    std::os::unix::net::UnixListener::bind(exchanged).unwrap();
                }
                "file" => fs::write(exchanged, b"public regular file").unwrap(),
                _ => unreachable!(),
            }
        });
        assert!(LocalContentProvider::new(&[root]).is_err(), "{kind}");
        assert!(fired.get(), "{kind}");
    }
}

#[test]
fn local_content_regular_file_replacement_still_fails_identity_comparison() {
    let (_temp, root) = fixture();
    let path = root.join("note.rs");
    fs::write(&path, b"public original").unwrap();
    let provider = LocalContentProvider::new(std::slice::from_ref(&root)).unwrap();
    let exchanged = path.clone();
    let fired = before_open(&path, move || {
        fs::rename(&exchanged, exchanged.with_file_name("retired.rs")).unwrap();
        fs::write(exchanged, b"public unwanted replacement").unwrap();
    });
    let error = provider
        .read(&ResourceId::from_path(path).unwrap())
        .unwrap_err();
    assert!(fired.get());
    assert!(matches!(error, crate::Error::Io { source, .. }
        if source.kind() == io::ErrorKind::Interrupted
            && source.to_string() == "local-content path changed while opening"));
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires scripts/test/local_content_autofs.c in an isolated privileged Linux mount namespace"]
fn local_content_real_cold_autofs() {
    let trigger = PathBuf::from(std::env::var_os("SEYK_AUTOFS_TRIGGER").expect("autofs helper"));
    let root = if std::env::var("SEYK_AUTOFS_ANCESTOR").unwrap() == "1" {
        trigger.join("selected")
    } else {
        trigger.clone()
    };
    // Policy preflight may warm an ancestor mount. Make it cold again before
    // open_directory reaches it, without replacing or mocking any syscall.
    let reset = trigger.clone();
    let reset_fired = before_open(trigger.parent().unwrap(), move || {
        let path = c_name(reset.as_os_str()).unwrap();
        // SAFETY: the helper owns this mount in a private test namespace.
        let result = unsafe { libc::umount2(path.as_ptr(), 0) };
        if result != 0 {
            assert_eq!(
                io::Error::last_os_error().raw_os_error(),
                Some(libc::EINVAL)
            );
        }
    });
    let cold = Rc::new(Cell::new(None));
    let observed = cold.clone();
    let component = trigger.clone();
    let fired = before_open(&trigger, move || {
        let parent = File::open(component.parent().unwrap()).unwrap();
        let stat = stat_at(&parent, component.file_name().unwrap()).unwrap();
        observed.set(Some(identity(&stat)));
    });
    let result = LocalContentProvider::new(std::slice::from_ref(&root));
    assert!(reset_fired.get());
    assert!(fired.get());
    let provider = result.unwrap();
    let mounted = identity(&descriptor_stat(&File::open(&trigger).unwrap()).unwrap());
    println!(
        "cold autofs stat dev:ino={:?}; opened tmpfs dev:ino={mounted:?}",
        cold.get().unwrap()
    );
    assert_ne!(cold.get().unwrap(), mounted);
    assert_content(&provider, &root);

    let outside = trigger.parent().unwrap().join("outside.rs");
    fs::write(&outside, b"public outside sentinel").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link.rs")).unwrap();
    let fifo = c_name(root.join("fifo.rs").as_os_str()).unwrap();
    // SAFETY: valid NUL-terminated fixture path, only creates a FIFO.
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    for name in ["link.rs", "fifo.rs"] {
        let id = ResourceId::from_path(root.join(name)).unwrap();
        assert!(provider.read(&id).is_err());
        assert!(provider.open(&id).is_err());
    }

    let path = c_name(trigger.as_os_str()).unwrap();
    // SAFETY: detach only this helper-owned backing mount, leaving autofs cold.
    assert_eq!(unsafe { libc::umount2(path.as_ptr(), libc::MNT_DETACH) }, 0);
    // Opening now obtains a newly mounted tmpfs, not the lifetime-pinned root.
    drop(File::open(&root).unwrap());
    let id = ResourceId::from_path(root.join("note.rs")).unwrap();
    assert!(provider.read(&id).is_err());
    assert!(provider.open(&id).is_err());
    assert!(provider.version(&id).is_err());
    assert!(provider.list(&scope(&root)).is_err());
    fs::remove_file(outside).unwrap();
}
