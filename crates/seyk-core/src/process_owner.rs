//! Process creation identity prevents PID reuse from making old markers live.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProcessOwner {
    pub(crate) pid: u32,
    pub(crate) start: String,
}

impl ProcessOwner {
    pub(crate) fn current() -> Option<Self> {
        let pid = std::process::id();
        Some(Self {
            pid,
            start: process_start(pid).ok()??,
        })
    }

    /// Unknown access is transient; only a known absent/different owner is dead.
    pub(crate) fn is_alive(&self) -> Option<bool> {
        Some(process_start(self.pid).ok()?.as_ref() == Some(&self.start))
    }
}

#[cfg(target_os = "macos")]
fn process_start(pid: u32) -> std::io::Result<Option<String>> {
    let Ok(pid) = i32::try_from(pid) else {
        return Ok(None);
    };
    let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::uninit();
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as i32;
    // SAFETY: proc_pidinfo writes at most the supplied POD buffer size.
    let read = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    if read != size {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(None)
        } else {
            Err(error)
        };
    }
    // SAFETY: the complete struct was initialized by proc_pidinfo.
    let info = unsafe { info.assume_init() };
    if info.pbi_status == libc::SZOMB {
        return Ok(None);
    }
    Ok(Some(format!(
        "{}:{}",
        info.pbi_start_tvsec, info.pbi_start_tvusec
    )))
}

#[cfg(target_os = "linux")]
fn process_start(pid: u32) -> std::io::Result<Option<String>> {
    let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let tail = stat
        .rsplit_once(')')
        .map(|(_, tail)| tail)
        .ok_or_else(|| std::io::Error::other("invalid process creation record"))?;
    if matches!(tail.split_whitespace().next(), Some("Z" | "X")) {
        return Ok(None);
    }
    let ticks = tail
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| std::io::Error::other("invalid process creation record"))?;
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    Ok(Some(format!("{}:{ticks}", boot.trim())))
}

#[cfg(windows)]
fn process_start(pid: u32) -> std::io::Result<Option<String>> {
    use windows_sys::Win32::{
        Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, FILETIME},
        System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
    };
    // SAFETY: the returned handle is queried read-only and closed below.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
            Ok(None)
        } else {
            Err(error)
        };
    }
    // SAFETY: FILETIME is a POD output record whose all-zero value is valid.
    let mut times: [FILETIME; 4] = unsafe { std::mem::zeroed() };
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: four distinct initialized FILETIME outputs and a valid process handle.
    let success = unsafe { GetProcessTimes(handle, creation, exit, kernel, user) };
    let error = std::io::Error::last_os_error();
    // SAFETY: this handle came from OpenProcess and is no longer used.
    unsafe { CloseHandle(handle) };
    if success == 0 {
        return Err(error);
    }
    if times[1].dwHighDateTime != 0 || times[1].dwLowDateTime != 0 {
        return Ok(None);
    }
    Ok(Some(format!(
        "{}:{}",
        times[0].dwHighDateTime, times[0].dwLowDateTime
    )))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn process_start(_: u32) -> std::io::Result<Option<String>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "process creation identity unavailable",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_identity_distinguishes_a_reused_pid() {
        let owner = ProcessOwner::current().unwrap();
        assert_eq!(owner.is_alive(), Some(true));
        let reused = ProcessOwner {
            start: format!("{}-old", owner.start),
            ..owner
        };
        assert_eq!(reused.is_alive(), Some(false));
    }
}
