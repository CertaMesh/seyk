use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .ok()?;
    output.status.success().then_some(())?;
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn version(root: &Path, package: &str) -> String {
    // Do not discover an unrelated enclosing repository around an unpacked crate.
    if !root.join(".git").exists() {
        return package.to_owned();
    }
    let Some(sha) = git(root, &["rev-parse", "--short", "HEAD"]) else {
        return package.to_owned();
    };
    // Cargo adds this untracked marker to its `cargo install --git`
    // checkout. It is installer metadata, not a source modification.
    let Some(status) = git(root, &["status", "--porcelain", "--untracked-files=normal"]) else {
        return package.to_owned();
    };
    let dirty = status.lines().any(|line| line != "?? .cargo-ok");
    let tags = git(root, &["tag", "--points-at", "HEAD"]).unwrap_or_default();
    let release = tags
        .lines()
        .any(|tag| tag == package || tag == format!("v{package}"));
    if release && !dirty {
        package.to_owned()
    } else {
        format!("{package}+g{sha}{}", if dirty { ".dirty" } else { "" })
    }
}

fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // Watch real source inputs, not target/ or an always-missing marker.
    // An untracked file outside these paths updates .dirty on the next rebuild.
    for name in [
        "src",
        "crates",
        "tests",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
    ] {
        watch(&root.join(name));
    }
    if root.join(".git").exists() {
        // Resolve paths through Git: linked worktrees have a .git *file*, a
        // private HEAD/index, and shared refs/packed-refs in the common git dir.
        for name in ["HEAD", "index", "refs", "packed-refs"] {
            if let Some(path) = git(&root, &["rev-parse", "--git-path", name]) {
                watch(&root.join(path));
            }
        }
    }
    println!(
        "cargo:rustc-env=SEYK_BUILD_VERSION={}",
        version(&root, env!("CARGO_PKG_VERSION"))
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_without_metadata_uses_package_version() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(version(dir.path(), "0.1.0"), "0.1.0");
    }

    #[test]
    fn inherited_git_environment_cannot_redirect_fixture_commands() {
        let sentinel = tempfile::tempdir().unwrap();
        if git(sentinel.path(), &["--version"]).is_none() {
            return;
        }
        assert!(git(sentinel.path(), &["init", "--quiet"]).is_some());
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "build_script::tests::release_development_dirty_and_detached_install_checkouts",
            ])
            .env("GIT_DIR", sentinel.path().join(".git"))
            .env("GIT_WORK_TREE", sentinel.path())
            .env("GIT_INDEX_FILE", sentinel.path().join(".git/index"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!sentinel.path().join(".git/index").exists());
        assert_eq!(
            std::fs::read_dir(sentinel.path().join(".git/refs/tags"))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn release_development_dirty_and_detached_install_checkouts() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"));
        if !source.join(".git").exists() {
            return; // Published crates and systems without Git still build/test.
        }
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        if git(root, &["--version"]).is_none() {
            return;
        }
        assert!(git(root, &["init", "--quiet"]).is_some());
        assert!(git(root, &["config", "tag.gpgSign", "false"]).is_some());
        // Reuse an existing commit: fixtures never create unsigned commits.
        assert!(
            git(
                root,
                &[
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    source.to_str().unwrap(),
                    "HEAD"
                ]
            )
            .is_some()
        );
        assert!(git(root, &["checkout", "--quiet", "--detach", "FETCH_HEAD"]).is_some());
        let sha = git(root, &["rev-parse", "--short", "HEAD"]).unwrap();
        let development = format!("0.1.0+g{sha}");
        assert_eq!(version(root, "0.1.0"), development);
        std::fs::write(root.join(".cargo-ok"), "").unwrap();
        assert_eq!(version(root, "0.1.0"), development);
        assert!(git(root, &["tag", "unrelated"]).is_some());
        assert_eq!(version(root, "0.1.0"), development);
        assert!(git(root, &["tag", "v0.1.0"]).is_some());
        assert_eq!(version(root, "0.1.0"), "0.1.0");
        std::fs::write(root.join("Cargo.toml"), "dirty\n").unwrap();
        assert_eq!(version(root, "0.1.0"), format!("{development}.dirty"));
        assert!(git(root, &["add", "Cargo.toml"]).is_some());
        assert_eq!(version(root, "0.1.0"), format!("{development}.dirty"));
        assert!(git(root, &["reset", "--hard", "--quiet", "HEAD"]).is_some());
        std::fs::write(root.join("untracked-source"), "dirty\n").unwrap();
        assert_eq!(version(root, "0.1.0"), format!("{development}.dirty"));
        std::fs::remove_file(root.join("untracked-source")).unwrap();
        assert!(git(root, &["tag", "-d", "v0.1.0"]).is_some());
        assert!(
            git(
                root,
                &[
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "tag",
                    "-a",
                    "-m",
                    "release",
                    "0.1.0"
                ]
            )
            .is_some()
        );
        assert_eq!(version(root, "0.1.0"), "0.1.0");
        let archive = root.join("unpacked-archive");
        std::fs::create_dir(&archive).unwrap();
        assert_eq!(version(&archive, "0.1.0"), "0.1.0");
        std::fs::remove_dir(&archive).unwrap();
        let linked_dir = tempfile::tempdir().unwrap();
        let linked = linked_dir.path().join("checkout");
        assert!(
            git(
                root,
                &[
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    linked.to_str().unwrap(),
                    "HEAD"
                ]
            )
            .is_some()
        );
        assert!(linked.join(".git").is_file());
        assert_eq!(version(&linked, "0.1.0"), "0.1.0");
        std::fs::write(linked.join("untracked-source"), "dirty\n").unwrap();
        assert_eq!(version(&linked, "0.1.0"), format!("{development}.dirty"));
        assert!(
            git(
                root,
                &["worktree", "remove", "--force", linked.to_str().unwrap()]
            )
            .is_some()
        );
    }
}
