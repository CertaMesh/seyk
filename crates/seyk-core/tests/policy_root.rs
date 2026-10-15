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
