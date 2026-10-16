use seyk_core::discovery::PolicyRoot;

#[test]
fn roots_require_existing_policy_allowed_paths() {
    let temp = tempfile::tempdir().unwrap();
    assert!(PolicyRoot::new(&temp.path().join("missing")).is_err());
    let secret = temp.path().join("id_rsa");
    std::fs::write(&secret, "secret").unwrap();
    assert!(matches!(
        PolicyRoot::new(&secret),
        Err(seyk_core::Error::SecretFile { .. })
    ));
}

#[cfg(unix)]
#[test]
fn chained_alias_retains_policy_context_and_retargeting_invalidates_identity() {
    use seyk_core::discovery::ensure_allowed_path;
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let store = base.join("store");
    let other = base.join("other");
    std::fs::create_dir(&store).unwrap();
    std::fs::create_dir(&other).unwrap();
    symlink("store", base.join(".docker")).unwrap();
    symlink(".docker", base.join("middle")).unwrap();
    let alias = base.join("alias");
    symlink("middle", &alias).unwrap();
    let root = PolicyRoot::new(&alias).unwrap();
    assert_eq!(root.uri_path(), alias);
    assert_eq!(root.authorize_path(), store);
    assert!(root.revalidate().unwrap());
    let logical = root.logical_child(&store.join("config.json")).unwrap();
    assert_eq!(logical, alias.join("config.json"));
    assert!(ensure_allowed_path(&logical).is_err());
    assert!(root.logical_child(&other.join("config.json")).is_none());
    std::fs::remove_file(&alias).unwrap();
    symlink("other", &alias).unwrap();
    assert!(!root.revalidate().unwrap());
    // Live observation must not overwrite the pinned session identity.
    assert_eq!(root.current().unwrap().authorize_path(), other);
    assert_eq!(root.authorize_path(), store);
    std::fs::remove_file(&alias).unwrap();
    assert!(root.current().is_err());
    assert!(root.revalidate().is_err());
}

#[test]
fn canonical_identity_uses_stored_case_without_rewriting_the_uri() {
    let temp = tempfile::tempdir().unwrap();
    let stored = temp.path().canonicalize().unwrap().join("scope");
    std::fs::create_dir(&stored).unwrap();
    let supplied = stored.with_file_name("SCOPE");
    if supplied.canonicalize().ok().as_ref() != Some(&stored) {
        return;
    }
    let root = PolicyRoot::new(&supplied).unwrap();
    assert_eq!(root.uri_path(), supplied);
    assert_eq!(root.authorize_path(), stored);
    assert!(root.revalidate().unwrap());
}

#[test]
fn logical_child_rejects_parent_traversal() {
    let temp = tempfile::tempdir().unwrap();
    let root = PolicyRoot::new(temp.path()).unwrap();
    for relative in ["../outside", "child/../../outside", "child/../sibling"] {
        let candidate = root.authorize_path().join(relative);
        assert!(root.logical_child(&candidate).is_none(), "{relative}");
        assert!(root.current().unwrap().logical_child(&candidate).is_none());
    }
    assert_eq!(
        root.logical_child(root.authorize_path()).unwrap(),
        root.uri_path()
    );
    assert_eq!(
        root.logical_child(&root.authorize_path().join("child/file.txt"))
            .unwrap(),
        root.uri_path().join("child/file.txt")
    );
}

#[cfg(unix)]
#[test]
fn live_observation_is_distinct_and_revalidation_rechecks_same_target_aliases() {
    use seyk_core::discovery::{LiveRoot, RootRevalidationError};
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let base = temp.path().canonicalize().unwrap();
    let store = base.join("store");
    std::fs::create_dir(&store).unwrap();
    let alias = base.join("alias");
    symlink("store", &alias).unwrap();
    let root = PolicyRoot::new(&alias).unwrap();
    assert!(root.revalidate().unwrap());

    symlink("store", base.join("vault.opvault")).unwrap();
    std::fs::remove_file(&alias).unwrap();
    symlink("vault.opvault", &alias).unwrap();
    let live: LiveRoot<'_> = root.current().unwrap();
    assert_eq!(live.uri_path(), alias);
    assert_eq!(live.authorize_path(), store);
    assert_eq!(root.authorize_path(), store);
    assert!(matches!(
        root.revalidate(),
        Err(RootRevalidationError::Policy(
            seyk_core::Error::SecretFile { path }
        )) if path == alias
    ));
    assert!(matches!(
        PolicyRoot::new(live.uri_path()),
        Err(seyk_core::Error::SecretFile { .. })
    ));
}

#[test]
fn revalidation_keeps_identity_io_errors_distinct() {
    use seyk_core::discovery::RootRevalidationError;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("root");
    std::fs::create_dir(&path).unwrap();
    let root = PolicyRoot::new(&path).unwrap();
    std::fs::remove_dir(&path).unwrap();
    assert!(matches!(
        root.revalidate(),
        Err(RootRevalidationError::Identity(error))
            if error.kind() == std::io::ErrorKind::NotFound
    ));
}

#[cfg(unix)]
#[test]
fn construction_preserves_filesystem_handling_of_directory_only_spellings() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("file.txt");
    std::fs::write(&file, "ordinary file").unwrap();
    for suffix in ["/", "/."] {
        let mut spelling = file.as_os_str().to_os_string();
        spelling.push(suffix);
        let path = std::path::PathBuf::from(spelling);
        match path.canonicalize() {
            Ok(expected) => {
                assert_eq!(PolicyRoot::new(&path).unwrap().authorize_path(), expected);
            }
            Err(expected) => {
                assert!(matches!(
                    PolicyRoot::new(&path),
                    Err(seyk_core::Error::Io { path: failed, source })
                        if failed == path && source.kind() == expected.kind()
                            && source.to_string() == expected.to_string()
                ));
            }
        }
    }
}
