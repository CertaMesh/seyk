mod common;

use std::fs;
use std::path::Path;

use seyk_core::cache::{ResourceWrite, SegmentWrite};
use seyk_core::discovery::{discover, is_secret_path};
use seyk_core::{
    Dimension, DiscoveryOptions, Error, FsProvider, MemoryCache, ResourceId, ResourceProvider,
    SearchEngine, SearchMode, SqliteCache,
};

fn permissive() -> DiscoveryOptions {
    DiscoveryOptions {
        include: vec!["*".into(), "**/*".into()],
        include_hidden: true,
        default_excludes: false,
        ..Default::default()
    }
}

#[test]
fn secret_matcher_covers_names_extensions_ancestors_and_case() {
    for name in [
        "key.pem",
        "key.key",
        "key.p12",
        "key.pfx",
        "key.jks",
        "key.keystore",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_xmss",
        "id_ed25519_sk",
        "id_rsa.backup",
        "id_rsa-copy",
        "id_rsa.txt",
        "vault.kdbx",
        "vault.kdb",
        "vault.1pif",
        "vault.agilekeychain/data.txt",
        "vault.opvault/data.md",
        ".env",
        ".env.local",
        ".env.example",
        ".env.local.toml",
        "prod.env",
        "export.keychain",
        "export.keychain-db",
        ".netrc",
        "_netrc",
        "netrc",
        ".pgpass",
        "pgpass",
        "pgpass.conf",
        ".aws/credentials",
        "credentials",
        "credentials.json",
        ".docker/config.json",
    ] {
        for variant in [name.to_string(), name.to_ascii_uppercase()] {
            assert!(is_secret_path(Path::new(&variant)), "{variant}");
            assert!(
                is_secret_path(&Path::new("notes").join(&variant)),
                "{variant}"
            );
        }
    }
    for name in [
        "notes.txt",
        "config.json",
        "docker/config.json",
        "id_rsa.pub",
        "id_ed25519_sk.pub",
        "my_id_rsa_notes.md",
        "id_rsausage.md",
        "environment.toml",
        "prod.env.md",
        "credentials-guide.md",
        "vault.opvault-notes/readme.md",
    ] {
        assert!(!is_secret_path(Path::new(name)), "{name}");
    }
}

#[test]
fn includes_hidden_and_disabled_defaults_cannot_override_secrets() {
    let temp = tempfile::tempdir().unwrap();
    for name in [
        "safe.md",
        ".env.local.toml",
        "id_rsa.txt",
        "secret.opvault/notes.md",
        "secret.agilekeychain/notes.txt",
        ".aws/credentials",
        ".docker/config.json",
    ] {
        let path = temp.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "synthetic secret marker").unwrap();
    }
    let resources = discover(&[temp.path().into()], &permissive()).unwrap();
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0].path.file_name().unwrap(), "safe.md");
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "deny"))
        .build()
        .unwrap();
    let report = engine
        .index()
        .paths([temp.path()])
        .discovery(permissive())
        .execute()
        .unwrap();
    assert_eq!(report.sync.resources_discovered, 1);
    for sync in [false, true] {
        let results = engine
            .search("marker")
            .paths([temp.path()])
            .discovery(permissive())
            .include("*")
            .sync(sync)
            .execute()
            .unwrap();
        assert_eq!(results.stats.sync.resources_discovered, 1);
        assert!(
            results.hits.iter().all(|hit| hit
                .resource
                .to_file_path()
                .unwrap()
                .ends_with("safe.md"))
        );
    }
}

#[test]
fn explicit_paths_and_file_uris_are_refused_before_reading() {
    let temp = tempfile::tempdir().unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "deny"))
        .build()
        .unwrap();
    for name in ["private.pem", "id_rsa", ".env", "vault.opvault/note.txt"] {
        let path = temp.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "synthetic marker").unwrap();
        let id = ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
        assert!(matches!(
            discover(std::slice::from_ref(&path), &permissive()),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            engine.index().paths([&path]).execute(),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            engine
                .search("marker")
                .roots([id.clone()])
                .sync(false)
                .execute(),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            FsProvider.read(&id),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            FsProvider.open(&id),
            Err(Error::SecretFile { .. })
        ));
    }
}

// Seed a legacy record through the cache API, without weakening discovery for a test.
fn assert_legacy_cache_is_not_served(cache: impl seyk_core::EmbeddingCache + 'static) {
    let temp = tempfile::tempdir().unwrap();
    let safe = temp.path().join("safe.txt");
    let secret = temp.path().join(".env.local.toml");
    let body = "synthetic quartz marker";
    fs::write(&safe, body).unwrap();
    fs::write(&secret, body).unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "legacy-deny"))
        .cache(cache)
        .build()
        .unwrap();
    let mut hit = engine
        .search("quartz")
        .paths([&safe])
        .execute()
        .unwrap()
        .hits
        .remove(0);
    let secret_id = ResourceId::from_path(secret.canonicalize().unwrap()).unwrap();
    engine
        .with_cache(|cache| {
            let old = cache.resource(engine.profile_id(), &hit.resource)?.unwrap();
            cache.put_resource(
                engine.profile_id(),
                &secret_id,
                &ResourceWrite {
                    metadata: &old.metadata,
                    version: &old.version,
                    descriptor_hash: old.descriptor_hash.as_deref(),
                    segments: old
                        .segments
                        .iter()
                        .map(|segment| SegmentWrite {
                            segment,
                            lexical_body: body,
                        })
                        .collect(),
                },
            )?;
            assert!(cache.resource(engine.profile_id(), &secret_id)?.is_some());
            Ok(())
        })
        .unwrap();
    for sync in [false, true] {
        for mode in [
            SearchMode::Semantic,
            SearchMode::Lexical,
            SearchMode::Hybrid,
        ] {
            let results = engine
                .search("quartz")
                .paths([temp.path()])
                .discovery(permissive())
                .mode(mode)
                .sync(sync)
                .previews(Some(100))
                .execute()
                .unwrap();
            assert!(!results.hits.is_empty());
            assert!(results.hits.iter().all(|hit| hit.resource != secret_id));
        }
    }
    hit.resource = secret_id;
    assert!(matches!(
        engine.extract(&hit),
        Err(Error::SecretFile { .. })
    ));
}

#[test]
fn legacy_memory_and_sqlite_entries_are_not_served() {
    assert_legacy_cache_is_not_served(MemoryCache::new());
    let temp = tempfile::tempdir().unwrap();
    assert_legacy_cache_is_not_served(SqliteCache::open(temp.path().join("cache.db")).unwrap());
}

#[cfg(unix)]
#[test]
fn symlink_alias_and_target_are_both_checked() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let safe = temp.path().join("safe.txt");
    let secret = temp.path().join("private.pem");
    fs::write(&safe, "safe").unwrap();
    fs::write(&secret, "synthetic secret").unwrap();
    let secret_alias = temp.path().join("id_rsa");
    let safe_alias = temp.path().join("alias.txt");
    symlink(&safe, &secret_alias).unwrap();
    symlink(&secret, &safe_alias).unwrap();
    for path in [&secret_alias, &safe_alias] {
        assert!(matches!(
            discover(&[path.into()], &permissive()),
            Err(Error::SecretFile { .. })
        ));
        let engine = SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(Dimension::D128, "alias"))
            .build()
            .unwrap();
        assert!(matches!(
            engine.search("safe").paths([path]).execute(),
            Err(Error::SecretFile { .. })
        ));
    }
}

#[test]
fn unicode_secret_spellings_are_normalized_on_every_platform() {
    for name in [
        "id_rſa.txt",
        "credentialſ",
        "private.Key",
        ".pgpaß",
        ".ｄｏｃｋｅｒ/CONFIG.JSON",
        "vault.ｏｐｖａｕｌｔ/note.txt",
        "id_rſa.PEM",
        "private.ＫＥＹ",
    ] {
        assert!(is_secret_path(Path::new(name)), "{name}");
    }
    for name in ["id_rſa.pub", "src/credentialſ/schema.rs", "monkey.txt"] {
        assert!(!is_secret_path(Path::new(name)), "{name}");
    }
}

#[test]
fn unicode_stored_names_are_refused_at_filesystem_entry_points() {
    use seyk_core::discovery::ensure_allowed_path;
    let temp = tempfile::tempdir().unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "unicode-deny"))
        .build()
        .unwrap();
    for (stored, alias) in [
        ("id_rſa.txt", "id_rsa.txt"),
        ("credentialſ", "credentials"),
        ("private.Key", "private.key"),
    ] {
        // Create the Unicode spelling FIRST: canonicalization must not hide the bug.
        let path = temp.path().join(stored);
        fs::write(&path, "synthetic quartz marker").unwrap();
        #[cfg(target_os = "macos")]
        if temp.path().join(alias).exists() {
            assert_eq!(
                fs::read(temp.path().join(alias)).unwrap(),
                fs::read(&path).unwrap()
            );
        }
        let _ = alias;
        assert!(matches!(
            ensure_allowed_path(&path),
            Err(Error::SecretFile { .. })
        ));
        let id = ResourceId::from_path(&path).unwrap();
        assert!(matches!(
            FsProvider.read(&id),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            FsProvider.open(&id),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            FsProvider.version(&id),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            engine.index().paths([&path]).execute(),
            Err(Error::SecretFile { .. })
        ));
        assert!(matches!(
            engine.search("quartz").roots([id]).sync(false).execute(),
            Err(Error::SecretFile { .. })
        ));
    }
    assert!(
        discover(&[temp.path().into()], &permissive())
            .unwrap()
            .is_empty()
    );
    assert!(
        engine
            .search("quartz")
            .paths([temp.path()])
            .discovery(permissive())
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
}

#[test]
fn ordinary_credentials_directories_remain_searchable() {
    let temp = tempfile::tempdir().unwrap();
    let folder = temp.path().join("src/credentials");
    fs::create_dir_all(&folder).unwrap();
    let path = folder.join("schema.rs");
    fs::write(&path, "pub struct LoginForm;").unwrap();
    seyk_core::discovery::ensure_allowed_path(&path).unwrap();
    seyk_core::discovery::ensure_allowed_path(&folder).unwrap();
    for root in [temp.path(), &folder, &path] {
        let found = discover(&[root.into()], &permissive()).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, path.canonicalize().unwrap());
    }
}

#[cfg(unix)]
#[test]
fn docker_symlink_root_preserves_context_and_filters_cached_results() {
    for chained in [false, true] {
        docker_symlink_root_preserves_context_and_filters_cached_results_case(chained);
    }
}

#[cfg(unix)]
fn docker_symlink_root_preserves_context_and_filters_cached_results_case(chained: bool) {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let store = temp.path().join("config-store");
    fs::create_dir(&store).unwrap();
    let config = store.join("config.json");
    fs::write(
        &config,
        r#"{"auths":{"example.invalid":{"auth":"synthetic quartz"}}}"#,
    )
    .unwrap();
    fs::write(store.join("safe.txt"), "synthetic quartz").unwrap();
    let docker = temp.path().join(".docker");
    symlink(&store, &docker).unwrap();
    let docker = if chained {
        let middle = temp.path().join("middle");
        let alias = temp.path().join("docker-alias");
        std::os::unix::fs::symlink(".docker", &middle).unwrap();
        std::os::unix::fs::symlink("middle", &alias).unwrap();
        alias
    } else {
        docker
    };
    fs::create_dir(store.join("sub")).unwrap();
    assert!(matches!(
        seyk_core::discovery::ensure_allowed_path(&docker.join("sub/../config.json")),
        Err(Error::SecretFile { .. })
    ));
    // An absolute child link re-enters the same physical parent using a new
    // spelling, then .. returns to it. The earlier .docker context still applies.
    fs::create_dir(store.join("real")).unwrap();
    symlink(store.join("real"), store.join("absolute-child")).unwrap();
    symlink("../config-store/real", store.join("relative-child")).unwrap();
    for child in ["absolute-child", "relative-child"] {
        let through_parent = docker.join(child).join("../config.json");
        assert!(
            matches!(
                seyk_core::discovery::ensure_allowed_path(&through_parent),
                Err(Error::SecretFile { .. })
            ),
            "{}",
            through_parent.display()
        );
    }
    let logical = docker.join("config.json");
    let id = ResourceId::from_path(&logical).unwrap();
    assert!(matches!(
        FsProvider.read(&id),
        Err(Error::SecretFile { .. })
    ));
    assert!(matches!(
        FsProvider.open(&id),
        Err(Error::SecretFile { .. })
    ));
    assert!(matches!(
        FsProvider.version(&id),
        Err(Error::SecretFile { .. })
    ));
    assert!(matches!(
        discover(std::slice::from_ref(&logical), &permissive()),
        Err(Error::SecretFile { .. })
    ));
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "docker-deny"))
        .build()
        .unwrap();
    // Simulate an old cache entry with the resolved identity, admitted through
    // the ordinary-name target. Both synced and cached-only alias searches filter it.
    let old = engine.search("quartz").paths([&config]).execute().unwrap();
    assert!(!old.hits.is_empty());
    engine
        .index()
        .paths([&docker])
        .discovery(permissive())
        .execute()
        .unwrap();
    for sync in [false, true] {
        for mode in [
            SearchMode::Semantic,
            SearchMode::Lexical,
            SearchMode::Hybrid,
        ] {
            for query in ["quartz", "config.json"] {
                let results = engine
                    .search(query)
                    .paths([&docker])
                    .discovery(permissive())
                    .mode(mode)
                    .sync(sync)
                    .execute()
                    .unwrap();
                assert!(
                    results.hits.iter().all(|hit| hit
                        .resource
                        .to_file_path()
                        .unwrap()
                        .ends_with("safe.txt"))
                );
                let results = engine
                    .search(query)
                    .roots([ResourceId::from_path(&docker).unwrap()])
                    .discovery(permissive())
                    .mode(mode)
                    .sync(sync)
                    .execute()
                    .unwrap();
                assert!(
                    results.hits.iter().all(|hit| hit
                        .resource
                        .to_file_path()
                        .unwrap()
                        .ends_with("safe.txt"))
                );
            }
        }
    }
    assert!(matches!(
        engine.search("quartz").paths([&logical]).execute(),
        Err(Error::SecretFile { .. })
    ));
    let mut old_hit = old.hits[0].clone();
    old_hit.resource = id;
    assert!(matches!(
        engine.extract(&old_hit),
        Err(Error::SecretFile { .. })
    ));
}

#[cfg(unix)]
#[test]
fn logical_docker_context_survives_parent_components_before_uri_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let store = temp.path().join("config-store");
    fs::create_dir_all(store.join("sub")).unwrap();
    fs::write(store.join("config.json"), "synthetic secret").unwrap();
    let docker = temp.path().join(".docker");
    std::os::unix::fs::symlink(&store, &docker).unwrap();
    let with_parent = docker.join("sub/../config.json");
    let id = ResourceId::from_path(
        seyk_core::discovery::absolute_logical_path(&docker)
            .unwrap()
            .join("config.json"),
    )
    .unwrap();
    assert!(matches!(
        seyk_core::discovery::ensure_allowed_path(&with_parent),
        Err(Error::SecretFile { .. })
    ));
    assert!(matches!(
        FsProvider.read(&id),
        Err(Error::SecretFile { .. })
    ));
    let logical = seyk_core::discovery::absolute_logical_path(&docker.join("sub/..")).unwrap();
    assert_eq!(logical, docker);
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "parents-deny"))
        .build()
        .unwrap();
    assert!(matches!(
        engine
            .index()
            .paths([docker.join("sub/../config.json")])
            .execute(),
        Err(Error::SecretFile { .. })
    ));
}

#[cfg(unix)]
#[test]
fn every_secret_symlink_hop_is_refused_across_entry_points() {
    use seyk_core::discovery::ensure_allowed_path;
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir_all(temp.path().join("store/sub")).unwrap();
    fs::create_dir(temp.path().join("links")).unwrap();
    let safe = temp.path().join("store/note.txt");
    fs::write(&safe, "synthetic quartz marker").unwrap();
    symlink("store", temp.path().join("vault.opvault")).unwrap();
    symlink("vault.opvault", temp.path().join("alias-dir")).unwrap();
    symlink("alias-dir", temp.path().join("middle")).unwrap();
    symlink("../middle", temp.path().join("links/outer")).unwrap();
    symlink("../vault.opvault/sub/..", temp.path().join("links/parents")).unwrap();
    symlink("store/note.txt", temp.path().join("id_rsa")).unwrap();
    symlink("id_rsa", temp.path().join("alias.txt")).unwrap();
    symlink("alias.txt", temp.path().join("outer.txt")).unwrap();
    symlink("store", temp.path().join("vault.ｏｐｖａｕｌｔ")).unwrap();
    symlink("vault.ｏｐｖａｕｌｔ", temp.path().join("unicode-alias")).unwrap();
    for cache in [false, true] {
        let builder = SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(Dimension::D128, "chain-deny"));
        let engine = if cache {
            builder
                .cache(SqliteCache::open(temp.path().join("cache.sqlite")).unwrap())
                .build()
                .unwrap()
        } else {
            builder.cache(MemoryCache::new()).build().unwrap()
        };
        let old = engine
            .search("quartz")
            .paths([&safe])
            .execute()
            .unwrap()
            .hits
            .remove(0);
        for spelling in [
            "alias-dir/note.txt",
            "alias.txt",
            "outer.txt",
            "links/outer/note.txt",
            "links/outer/sub/../note.txt",
            "links/parents/note.txt",
            "unicode-alias/note.txt",
        ] {
            let path = temp.path().join(spelling);
            assert_eq!(fs::read(&path).unwrap(), fs::read(&safe).unwrap());
            let id = ResourceId::from_path(&path).unwrap();
            assert!(
                matches!(ensure_allowed_path(&path), Err(Error::SecretFile { .. })),
                "{spelling}"
            );
            assert!(matches!(
                discover(std::slice::from_ref(&path), &permissive()),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                FsProvider.read(&id),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                FsProvider.open(&id),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                FsProvider.version(&id),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                engine.index().paths([&path]).execute(),
                Err(Error::SecretFile { .. })
            ));
            for mode in [
                SearchMode::Semantic,
                SearchMode::Lexical,
                SearchMode::Hybrid,
            ] {
                for sync in [false, true] {
                    for query in ["quartz", "note.txt"] {
                        assert!(matches!(
                            engine
                                .search(query)
                                .paths([&path])
                                .mode(mode)
                                .sync(sync)
                                .execute(),
                            Err(Error::SecretFile { .. })
                        ));
                        assert!(matches!(
                            engine
                                .search(query)
                                .roots([id.clone()])
                                .mode(mode)
                                .sync(sync)
                                .execute(),
                            Err(Error::SecretFile { .. })
                        ));
                    }
                }
            }
            let mut hit = old.clone();
            hit.resource = id;
            assert!(matches!(
                engine.extract(&hit),
                Err(Error::SecretFile { .. })
            ));
        }
        for spelling in ["alias-dir", "links/outer", "links/parents"] {
            let path = temp.path().join(spelling);
            #[cfg(feature = "watch")]
            assert!(matches!(
                seyk_core::watch::watch(
                    &engine,
                    std::slice::from_ref(&path),
                    &permissive(),
                    &seyk_core::watch::WatchOptions::default(),
                    None,
                    |_| panic!("secret chain must fail before any indexing pass"),
                ),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                discover(std::slice::from_ref(&path), &permissive()),
                Err(Error::SecretFile { .. })
            ));
            assert!(matches!(
                engine.search("quartz").paths([&path]).sync(false).execute(),
                Err(Error::SecretFile { .. })
            ));
        }
    }
}

#[cfg(unix)]
#[test]
fn resolver_bounds_loops_and_preserves_safe_filesystem_parent_semantics() {
    use seyk_core::discovery::{absolute_logical_path, ensure_allowed_path};
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    symlink("loop-b", temp.path().join("loop-a")).unwrap();
    symlink("loop-a", temp.path().join("loop-b")).unwrap();
    for path in [
        temp.path().join("loop-a"),
        temp.path().join("loop-a/note.txt"),
    ] {
        let err = ensure_allowed_path(&path).unwrap_err();
        assert!(matches!(err, Error::Io { .. }));
        assert!(err.to_string().contains("maximum 40"));
        assert!(discover(&[path], &permissive()).is_err());
    }
    fs::create_dir_all(temp.path().join("real/deep")).unwrap();
    fs::write(temp.path().join("real/note.txt"), "safe").unwrap();
    symlink("real/deep", temp.path().join("alias")).unwrap();
    let path = temp.path().join("alias/../note.txt");
    ensure_allowed_path(&path).unwrap();
    assert_eq!(
        absolute_logical_path(&path).unwrap(),
        path.canonicalize().unwrap()
    );
    // A contextual name must also survive a link that resolves to filesystem root.
    symlink("/", temp.path().join(".docker")).unwrap();
    assert!(matches!(
        ensure_allowed_path(&temp.path().join(".docker/config.json")),
        Err(Error::SecretFile { .. })
    ));
}

#[cfg(unix)]
#[test]
fn ordinary_directory_aliases_remain_searchable() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("store")).unwrap();
    std::fs::write(temp.path().join("store/schema.rs"), "struct LoginForm;").unwrap();
    std::os::unix::fs::symlink("store", temp.path().join("credentials")).unwrap();
    std::os::unix::fs::symlink("credentials", temp.path().join("alias")).unwrap();
    let path = temp.path().join("alias/schema.rs");
    seyk_core::discovery::ensure_allowed_path(&path).unwrap();
    assert_eq!(
        discover(&[temp.path().join("alias")], &permissive())
            .unwrap()
            .len(),
        1
    );
}
