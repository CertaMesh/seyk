use std::{ffi::OsString, path::PathBuf};

pub fn fixture() -> Result<PathBuf, &'static str> {
    resolve(|name| std::env::var_os(name))
}

fn resolve(env: impl Fn(&str) -> Option<OsString>) -> Result<PathBuf, &'static str> {
    let home = || {
        env("HOME")
            .or_else(|| env("USERPROFILE"))
            .map(PathBuf::from)
            .ok_or("set HOME, USERPROFILE or SEYK_PHOTO_FIXTURE")
    };
    if let Some(path) = env("SEYK_PHOTO_FIXTURE") {
        let path = PathBuf::from(path);
        return match path.strip_prefix("~") {
            Ok(relative) => Ok(home()?.join(relative)),
            Err(_) => Ok(path),
        };
    }
    Ok(home()?.join(".cache/seyk-bench/photo-fixture"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env(values: &[(&str, &str)]) -> Result<PathBuf, &'static str> {
        resolve(|key| {
            values
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| OsString::from(value))
        })
    }

    #[test]
    fn home_precedes_userprofile() {
        assert_eq!(
            with_env(&[("HOME", "unix-home"), ("USERPROFILE", "windows-home")]).unwrap(),
            PathBuf::from("unix-home/.cache/seyk-bench/photo-fixture")
        );
    }

    #[test]
    fn userprofile_is_used_when_home_is_unset() {
        assert_eq!(
            with_env(&[("USERPROFILE", "windows-home")]).unwrap(),
            PathBuf::from("windows-home/.cache/seyk-bench/photo-fixture")
        );
        assert_eq!(
            with_env(&[
                ("USERPROFILE", "windows-home"),
                ("SEYK_PHOTO_FIXTURE", "~/photos")
            ])
            .unwrap(),
            PathBuf::from("windows-home/photos")
        );
    }

    #[test]
    fn explicit_destination_needs_no_home() {
        assert_eq!(
            with_env(&[("SEYK_PHOTO_FIXTURE", "custom-fixture")]).unwrap(),
            PathBuf::from("custom-fixture")
        );
        assert!(with_env(&[]).is_err());
        assert!(with_env(&[("SEYK_PHOTO_FIXTURE", "~/photos")]).is_err());
    }
}
