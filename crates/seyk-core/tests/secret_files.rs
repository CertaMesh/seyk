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
