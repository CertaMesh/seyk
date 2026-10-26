//! One background process per configuration that every agent session shares.
//!
//! `seyk mcp` is a thin relay: it connects to the daemon's Unix socket (starting
//! the daemon if none is running), sends one hello line naming its folders, then
//! copies MCP bytes between the agent's stdio and the socket. The daemon holds the
//! only copy of the model and one watcher per folder, so five agent sessions cost
//! one model's memory instead of five. It exits after a quiet period with no
//! sessions, so the memory is only held while agents use it.
//!
//! The socket name hashes the build version, JSON and cache schema versions,
//! and model options. Mismatched clients start a separate daemon; existing
//! sessions keep their daemon until normal idle shutdown. Equal identity values
//! share an endpoint even when the implementation differs.

use std::fs::{File, OpenOptions};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use rmcp::ServiceExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::mcp::{EngineFactory, Hub, IndexStatus, SeykServer};

/// Where one configuration's daemon lives.
pub struct Endpoint {
    pub socket: PathBuf,
    private_socket_dir: bool,
    lock: PathBuf,
    log: PathBuf,
    /// Arguments that start a daemon with this configuration.
    spawn_args: Vec<String>,
}

impl Endpoint {
    /// `model_args` are the model flags, exactly as they would be passed to
    /// `seyk daemon run`. They and the version decide which daemon to use.
    pub fn new(model_args: Vec<String>) -> Result<Self> {
        let override_dir = std::env::var_os("SEYK_DAEMON_DIR");
        let dir = match &override_dir {
            Some(d) => PathBuf::from(d),
            None => dirs::cache_dir()
                .ok_or_else(|| anyhow!("no cache directory on this platform"))?
                .join("seyk"),
        };
        let id = endpoint_id(
            &model_args,
            env!("SEYK_BUILD_VERSION"),
            crate::JSON_SCHEMA_VERSION,
            seyk::cache::SCHEMA_VERSION,
        );
        let normal_socket = dir.join(format!("daemon-{id}.sock"));
        let socket = select_socket_path(
            &normal_socket,
            override_dir.is_some(),
            std::env::var_os("TMPDIR").as_deref().map(Path::new),
            Path::new("/tmp"),
            current_uid(),
            socket_path_limit(),
        )?;
        let private_socket_dir = socket != normal_socket;
        if private_socket_dir {
            ensure_socket_dir(socket.parent().unwrap(), current_uid())?;
        }
        std::fs::create_dir_all(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let mut spawn_args = vec!["daemon".to_string(), "run".to_string()];
        spawn_args.extend(model_args);
        Ok(Self {
            socket,
            private_socket_dir,
            lock: dir.join(format!("daemon-{id}.lock")),
            log: dir.join("daemon.log"),
            spawn_args,
        })
    }

    fn verify_socket_dir(&self) -> Result<()> {
        if self.private_socket_dir {
            verify_socket_dir(self.socket.parent().unwrap(), current_uid())?;
        }
        Ok(())
    }

    fn lock_conflict_error(&self) -> anyhow::Error {
        anyhow!(
            "another daemon holds the lock for this cache dir ({}), probably started with a different TMPDIR; no socket at {}; set SEYK_DAEMON_DIR to a common short directory",
            self.lock.parent().unwrap().display(),
            self.socket.display()
        )
    }

    fn check_lock_for_missing_socket(&self) -> Result<()> {
        if self.socket.exists() {
            return Ok(());
        }
        let lock = match OpenOptions::new().write(true).open(&self.lock) {
            Ok(lock) => lock,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("checking daemon endpoint lock"),
        };
        match lock.try_lock() {
            Ok(()) => Ok(()),
            Err(std::fs::TryLockError::WouldBlock) => Err(self.lock_conflict_error()),
            Err(e) => Err(e).context("checking daemon endpoint lock"),
        }
    }

    fn remove_socket(&self) -> Result<()> {
        self.verify_socket_dir()?;
        match std::fs::remove_file(&self.socket) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("removing {}", self.socket.display())),
        }
    }
}

fn current_uid() -> u32 {
    // geteuid has no preconditions and identifies the owner of newly created files.
    unsafe { libc::geteuid() }
}

fn socket_path_limit() -> usize {
    // All-zero bytes are valid for this C address struct. Use the platform's
    // actual array size (104 on macOS, 108 on Linux), including the trailing NUL.
    let addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_path.len()
}

fn check_socket_path(path: &Path, limit: usize) -> Result<()> {
    let len = path.as_os_str().as_bytes().len();
    if len >= limit {
        bail!(
            "daemon socket path {} is {len} bytes; AF_UNIX requires fewer than {limit} bytes (including room for a trailing NUL); set SEYK_DAEMON_DIR to a shorter directory",
            path.display()
        );
    }
    Ok(())
}

fn select_socket_path(
    normal: &Path,
    overridden: bool,
    tmpdir: Option<&Path>,
    fallback: &Path,
    uid: u32,
    limit: usize,
) -> Result<PathBuf> {
    if overridden || check_socket_path(normal, limit).is_ok() {
        check_socket_path(normal, limit)?;
        return Ok(normal.into());
    }
    // Include the cache path: distinct lock directories must never own the same
    // fallback socket. Retain 128 hash bits even with a short socket filename.
    let id = blake3::hash(normal.as_os_str().as_bytes()).to_hex();
    let name = format!("d-{}.sock", &id[..32]);
    let socket_in = |base: &Path| base.join(format!("seyk-{uid}")).join(&name);
    if let Some(tmpdir) = tmpdir.filter(|p| p.is_absolute()) {
        let socket = socket_in(tmpdir);
        if check_socket_path(&socket, limit).is_ok() {
            return Ok(socket);
        }
    }
    let socket = socket_in(fallback);
    check_socket_path(&socket, limit)?;
    Ok(socket)
}

fn ensure_socket_dir(dir: &Path, uid: u32) -> Result<()> {
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| {
                format!(
                    "setting private daemon socket directory mode {}",
                    dir.display()
                )
            })?,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => {
            return Err(e).with_context(|| {
                format!("creating private daemon socket directory {}", dir.display())
            });
        }
    }
    verify_socket_dir(dir, uid)
}

fn verify_socket_dir(dir: &Path, uid: u32) -> Result<()> {
    let meta = std::fs::symlink_metadata(dir)
        .with_context(|| format!("checking private daemon socket directory {}", dir.display()))?;
    if !meta.is_dir() || meta.uid() != uid || meta.mode() & 0o7777 != 0o700 {
        bail!(
            "refusing daemon socket directory {}: must be a real directory (not a symlink), owned by uid {uid}, with mode 0700; found uid {} and mode {:04o}",
            dir.display(),
            meta.uid(),
            meta.mode() & 0o7777
        );
    }
    Ok(())
}

fn endpoint_id(
    model_args: &[String],
    build_version: &str,
    schema_version: u32,
    cache_schema_version: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"daemon-build");
    hasher.update(&(build_version.len() as u64).to_le_bytes());
    hasher.update(build_version.as_bytes());
    hasher.update(b"cache-schema");
    hasher.update(&(cache_schema_version.len() as u64).to_le_bytes());
    hasher.update(cache_schema_version.as_bytes());
    // A new client must not reconnect to a daemon returning an older JSON shape.
    hasher.update(b"json-schema");
    hasher.update(&schema_version.to_le_bytes());
    for a in model_args {
        hasher.update(&(a.len() as u64).to_le_bytes());
        hasher.update(a.as_bytes());
    }
    hasher.finalize().to_hex()[..12].to_string()
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Hello {
    /// Serve MCP for these folders on the rest of the connection.
    Session {
        roots: Vec<PathBuf>,
    },
    Status,
    Stop,
}

#[derive(Debug, Serialize, Deserialize)]
struct Reply {
    ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    status: Option<DaemonStatus>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub pid: u32,
    pub version: String,
    pub sessions: usize,
    pub folders: Vec<(PathBuf, IndexStatus)>,
    pub cache_status: Option<seyk::CacheStats>,
    #[serde(default)]
    pub index_progress: Vec<super::mcp::RootProgress>,
}

// ---- daemon side -----------------------------------------------------------

/// Admission and idle shutdown share one lock: either a session owns a lease,
/// or closing wins and every later admission is refused.
struct Sessions {
    state: Mutex<SessionState>,
}

struct SessionState {
    active: usize,
    last_release: Instant,
    closing: bool,
}

impl Sessions {
    fn new(now: Instant) -> Self {
        Self {
            state: Mutex::new(SessionState {
                active: 0,
                last_release: now,
                closing: false,
            }),
        }
    }

    fn acquire(self: &Arc<Self>) -> Result<SessionLease> {
        let mut state = self.state.lock().unwrap();
        if state.closing {
            bail!("daemon is closing");
        }
        state.active += 1;
        Ok(SessionLease(self.clone()))
    }

    fn active(&self) -> usize {
        self.state.lock().unwrap().active
    }

    fn close_if_idle(&self, now: Instant, idle: Duration) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.active == 0 && now.saturating_duration_since(state.last_release) >= idle {
            state.closing = true;
        }
        state.closing
    }
}

struct SessionLease(Arc<Sessions>);

impl Drop for SessionLease {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.active -= 1;
        if state.active == 0 {
            state.last_release = Instant::now();
        }
    }
}

/// Run the daemon in the foreground until it is stopped or idle for `idle`.
pub fn run(endpoint: &Endpoint, factory: EngineFactory, idle: Duration) -> Result<()> {
    // One daemon per endpoint: whoever holds the lock owns the socket path.
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(&endpoint.lock)?;
    if lock.try_lock().is_err() {
        if !endpoint.socket.exists() {
            return Err(endpoint.lock_conflict_error());
        }
        eprintln!("seyk daemon: already running for this configuration");
        return Ok(());
    }
    // Holding the lock means any socket file left here is stale.
    endpoint.remove_socket()?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = rt.block_on(serve(endpoint, Hub::new(factory, true), idle));
    finish_shutdown(endpoint, rt, lock, result)
}

fn finish_shutdown(
    endpoint: &Endpoint,
    runtime: tokio::runtime::Runtime,
    lock: File,
    result: Result<()>,
) -> Result<()> {
    // Runtime drop waits for outstanding spawn_blocking work. Keep endpoint
    // ownership until it finishes so stop/restart cannot overlap that work.
    drop(runtime);
    let cleanup = endpoint.remove_socket();
    drop(lock);
    result.and(cleanup)
}

async fn serve(endpoint: &Endpoint, hub: Arc<Hub>, idle: Duration) -> Result<()> {
    endpoint.verify_socket_dir()?;
    let listener = UnixListener::bind(&endpoint.socket)
        .with_context(|| format!("binding {}", endpoint.socket.display()))?;
    std::fs::set_permissions(&endpoint.socket, std::fs::Permissions::from_mode(0o600))?;
    eprintln!(
        "seyk daemon {} (pid {}) listening on {}",
        env!("SEYK_BUILD_VERSION"),
        std::process::id(),
        endpoint.socket.display()
    );

    let sessions = Arc::new(Sessions::new(Instant::now()));
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    let mut connections = tokio::task::JoinSet::new();
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            conn = listener.accept() => {
                let (stream, _) = conn?;
                let (hub, sessions, stop) = (hub.clone(), sessions.clone(), stop_tx.clone());
                connections.spawn(async move {
                    if let Err(e) = handle(stream, hub, sessions, stop).await {
                        eprintln!("seyk daemon: connection: {e:#}");
                    }
                });
            }
            _ = connections.join_next(), if !connections.is_empty() => {}
            _ = stop_rx.recv() => {
                eprintln!("seyk daemon: stopped on request");
                return Ok(());
            }
            _ = tick.tick() => {
                if sessions.close_if_idle(Instant::now(), idle) {
                    eprintln!("seyk daemon: idle for {}s, exiting", idle.as_secs());
                    break;
                }
            }
        }
    }
    drop(listener);
    // Let accepted connections receive the closing reply before the runtime
    // exits. Incomplete hellos must not keep an idle daemon alive indefinitely.
    let _ = tokio::time::timeout(Duration::from_secs(1), async {
        while connections.join_next().await.is_some() {}
    })
    .await;
    Ok(())
}

async fn handle(
    stream: UnixStream,
    hub: Arc<Hub>,
    sessions: Arc<Sessions>,
    stop: tokio::sync::mpsc::Sender<()>,
) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let mut read = BufReader::new(read);
    let mut line = String::new();
    read.read_line(&mut line).await?;
    let hello: Hello = serde_json::from_str(&line).context("bad hello")?;

    match hello {
        Hello::Status => {
            let status = DaemonStatus {
                pid: std::process::id(),
                version: env!("SEYK_BUILD_VERSION").into(),
                sessions: sessions.active(),
                folders: hub.watched(),
                index_progress: hub.root_progress(None),
                cache_status: hub.cache_status().ok().flatten(),
            };
            reply(&mut write, Ok(Some(status))).await
        }
        Hello::Stop => {
            reply(&mut write, Ok(None)).await?;
            let _ = stop.send(()).await;
            Ok(())
        }
        Hello::Session { roots } => {
            let _lease = match sessions.acquire() {
                Ok(lease) => lease,
                Err(e) => return reply(&mut write, Err(e)).await,
            };
            // Building a session canonicalizes roots and may start watchers.
            let server =
                match tokio::task::spawn_blocking(move || SeykServer::new(roots, hub)).await? {
                    Ok(s) => s,
                    Err(e) => return reply(&mut write, Err(e)).await,
                };
            reply(&mut write, Ok(None)).await?;
            // The buffered reader keeps any MCP bytes that arrived with the hello.
            let service = server.serve((read, write)).await?;
            service.waiting().await?;
            Ok(())
        }
    }
}

async fn reply(
    write: &mut tokio::net::unix::OwnedWriteHalf,
    outcome: Result<Option<DaemonStatus>>,
) -> Result<()> {
    let r = match outcome {
        Ok(status) => Reply {
            ok: true,
            error: None,
            status,
        },
        Err(e) => Reply {
            ok: false,
            error: Some(format!("{e:#}")),
            status: None,
        },
    };
    let mut line = serde_json::to_string(&r)?;
    line.push('\n');
    write.write_all(line.as_bytes()).await?;
    write.flush().await?;
    Ok(())
}

// ---- client side -----------------------------------------------------------

/// Relay this process's stdio to a daemon session for `roots`, starting the
/// daemon if needed. Returns when either side closes.
pub fn relay(endpoint: &Endpoint, roots: Vec<PathBuf>) -> Result<()> {
    // Absolute, but retain logical directory names for the secret-file policy.
    let roots = roots
        .iter()
        .map(|r| {
            seyk::discovery::ensure_allowed_path(r)?;
            r.canonicalize()
                .with_context(|| format!("root {}", r.display()))?;
            seyk::discovery::PolicyRoot::new(r)
                .map(|root| root.uri_path().to_path_buf())
                .with_context(|| format!("root {}", r.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = rt.block_on(async {
        let stream = connect_or_spawn(endpoint).await?;
        let (read, mut write) = stream.into_split();
        let mut read = BufReader::new(read);
        let _ = hello(&mut read, &mut write, &Hello::Session { roots }).await?;

        // The agent closing stdin only half-closes the socket: replies to its
        // last requests are still on the way, so the session ends when the
        // daemon closes its side.
        tokio::spawn(async move {
            let _ = tokio::io::copy(&mut tokio::io::stdin(), &mut write).await;
            let _ = write.shutdown().await;
        });
        let mut out = tokio::io::stdout();
        tokio::io::copy(&mut read, &mut out).await?;
        out.flush().await?;
        anyhow::Ok(())
    });
    // A stdin read may still be blocked; do not wait for it.
    rt.shutdown_background();
    result
}

/// Ask a running daemon for its status, or `None` when none is running.
pub fn status(endpoint: &Endpoint) -> Result<Option<DaemonStatus>> {
    request(endpoint, &Hello::Status).map(|r| r.and_then(|r| r.status))
}

/// Stop a running daemon. Returns whether one was running.
pub fn stop(endpoint: &Endpoint) -> Result<bool> {
    let Some(_) = request(endpoint, &Hello::Stop)? else {
        return Ok(false);
    };
    // Wait for it to release the socket, so a following start gets a new one.
    let deadline = Instant::now() + Duration::from_secs(10);
    while endpoint.socket.exists() {
        if Instant::now() >= deadline {
            bail!("the daemon acknowledged but has not exited after 10 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(true)
}

fn request(endpoint: &Endpoint, msg: &Hello) -> Result<Option<Reply>> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let Ok(stream) = UnixStream::connect(&endpoint.socket).await else {
            return Ok(None);
        };
        let (read, mut write) = stream.into_split();
        let mut read = BufReader::new(read);
        hello(&mut read, &mut write, msg).await.map(Some)
    })
}

async fn hello(
    read: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
    write: &mut tokio::net::unix::OwnedWriteHalf,
    msg: &Hello,
) -> Result<Reply> {
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    write.write_all(line.as_bytes()).await?;
    let mut answer = String::new();
    read.read_line(&mut answer).await?;
    let reply: Reply = serde_json::from_str(&answer)
        .with_context(|| format!("unexpected daemon reply {answer:?}"))?;
    if !reply.ok {
        bail!("{}", reply.error.unwrap_or_else(|| "daemon refused".into()));
    }
    Ok(reply)
}

async fn connect_or_spawn(endpoint: &Endpoint) -> Result<UnixStream> {
    if let Ok(s) = UnixStream::connect(&endpoint.socket).await {
        return Ok(s);
    }
    spawn(endpoint)?;
    // The model loads lazily, so the socket appears within moments of starting.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        match UnixStream::connect(&endpoint.socket).await {
            Ok(s) => return Ok(s),
            Err(e) if Instant::now() >= deadline => {
                endpoint.check_lock_for_missing_socket()?;
                return Err(anyhow!(
                    "seyk daemon did not start ({e}); see {}",
                    endpoint.log.display()
                ));
            }
            Err(_) => {}
        }
    }
}

/// Start `seyk daemon run` detached from this session: its own process group,
/// no terminal, output appended to the log. If two sessions race, the lock
/// makes the second daemon exit at once.
fn spawn(endpoint: &Endpoint) -> Result<()> {
    use std::os::unix::process::CommandExt;
    let log = log_file(&endpoint.log)?;
    Command::new(std::env::current_exe()?)
        .args(&endpoint.spawn_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .process_group(0)
        .spawn()
        .context("starting seyk daemon")?;
    Ok(())
}

fn log_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socket_selection_keeps_fitting_paths_and_explicit_override() {
        let normal = Path::new("/cache/seyk/daemon-123456789abc.sock");
        for overridden in [false, true] {
            assert_eq!(
                select_socket_path(
                    normal,
                    overridden,
                    Some(Path::new("/temp")),
                    Path::new("/tmp"),
                    42,
                    socket_path_limit()
                )
                .unwrap(),
                normal
            );
        }
    }

    #[test]
    fn socket_selection_prefers_tmpdir_then_short_fallback() {
        let normal = PathBuf::from(format!("/{}/daemon-123456789abc.sock", "long".repeat(40)));
        let choose = |tmpdir| {
            select_socket_path(
                &normal,
                false,
                tmpdir,
                Path::new("/tmp"),
                42,
                socket_path_limit(),
            )
            .unwrap()
        };
        let preferred = choose(Some(Path::new("/temp")));
        assert_eq!(preferred.parent().unwrap(), Path::new("/temp/seyk-42"));
        let name = preferred.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("d-") && name.ends_with(".sock"));
        assert_eq!(name.len(), 2 + 32 + 5);
        let long_tmp = PathBuf::from(format!("/{}", "temp".repeat(40)));
        for tmpdir in [Some(long_tmp.as_path()), None] {
            let short = choose(tmpdir);
            assert_eq!(short.parent().unwrap(), Path::new("/tmp/seyk-42"));
            assert_eq!(short.file_name(), preferred.file_name());
        }
        let other = select_socket_path(
            &normal.with_file_name("daemon-other.sock"),
            false,
            None,
            Path::new("/tmp"),
            42,
            socket_path_limit(),
        )
        .unwrap();
        assert_ne!(other.file_name(), preferred.file_name());
        let other_cache = select_socket_path(
            &normal
                .parent()
                .unwrap()
                .join("other-cache/daemon-123456789abc.sock"),
            false,
            None,
            Path::new("/tmp"),
            42,
            socket_path_limit(),
        )
        .unwrap();
        assert_ne!(other_cache.file_name(), preferred.file_name());
    }

    #[test]
    fn socket_selection_ignores_empty_tmpdir() {
        let normal = PathBuf::from(format!("/{}", "long".repeat(40)));
        let socket = select_socket_path(
            &normal,
            false,
            Some(Path::new("")),
            Path::new("/tmp"),
            42,
            socket_path_limit(),
        )
        .unwrap();
        assert_eq!(socket.parent().unwrap(), Path::new("/tmp/seyk-42"));
    }

    #[test]
    fn socket_selection_ignores_relative_tmpdir() {
        let normal = PathBuf::from(format!("/{}", "long".repeat(40)));
        let socket = select_socket_path(
            &normal,
            false,
            Some(Path::new("tmp")),
            Path::new("/tmp"),
            42,
            socket_path_limit(),
        )
        .unwrap();
        assert_eq!(socket.parent().unwrap(), Path::new("/tmp/seyk-42"));
    }

    #[test]
    fn held_lock_without_expected_socket_explains_tmpdir_conflict() {
        let temp = tempfile::tempdir().unwrap();
        let endpoint = Endpoint {
            socket: temp.path().join("missing.sock"),
            private_socket_dir: false,
            lock: temp.path().join("daemon.lock"),
            log: temp.path().join("daemon.log"),
            spawn_args: vec![],
        };
        endpoint.check_lock_for_missing_socket().unwrap();
        let lock = File::create(&endpoint.lock).unwrap();
        lock.try_lock().unwrap();
        let error = endpoint
            .check_lock_for_missing_socket()
            .unwrap_err()
            .to_string();
        for expected in [
            "another daemon holds the lock for this cache dir",
            "different TMPDIR",
            "SEYK_DAEMON_DIR",
        ] {
            assert!(error.contains(expected), "{error}");
        }
        let error = run(&endpoint, Box::new(|| bail!("must not load a model")), IDLE)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different TMPDIR"));
        let _socket = std::os::unix::net::UnixListener::bind(&endpoint.socket).unwrap();
        endpoint.check_lock_for_missing_socket().unwrap();
        std::fs::remove_file(&endpoint.socket).unwrap();
        lock.unlock().unwrap();
        drop(lock);
        endpoint.check_lock_for_missing_socket().unwrap();
    }

    #[test]
    fn shutdown_keeps_socket_and_lock_until_blocking_work_finishes() {
        for failed in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let endpoint = Arc::new(Endpoint {
                socket: temp.path().join("daemon.sock"),
                private_socket_dir: false,
                lock: temp.path().join("daemon.lock"),
                log: temp.path().join("daemon.log"),
                spawn_args: vec![],
            });
            let lock = File::create(&endpoint.lock).unwrap();
            lock.try_lock().unwrap();
            let contender = OpenOptions::new().write(true).open(&endpoint.lock).unwrap();
            let listener = std::os::unix::net::UnixListener::bind(&endpoint.socket).unwrap();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (release, gate) = std::sync::mpsc::channel();
            let (started, ready) = std::sync::mpsc::channel();
            rt.spawn_blocking(move || {
                started.send(()).unwrap();
                // Dropping release on a test failure also unblocks teardown.
                let _ = gate.recv();
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            drop(listener);
            let (stopping, shutdown_started) = std::sync::mpsc::channel();
            let (finished, shutdown_finished) = std::sync::mpsc::channel();
            let shutdown = std::thread::spawn({
                let endpoint = endpoint.clone();
                move || {
                    let result = if failed {
                        Err(anyhow!("serve failed"))
                    } else {
                        Ok(())
                    };
                    stopping.send(()).unwrap();
                    let result = finish_shutdown(&endpoint, rt, lock, result);
                    finished.send(result).unwrap();
                }
            });
            shutdown_started
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            assert!(matches!(
                shutdown_finished.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            assert!(endpoint.socket.exists());
            assert!(matches!(
                contender.try_lock(),
                Err(std::fs::TryLockError::WouldBlock)
            ));
            release.send(()).unwrap();
            let result = shutdown_finished
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            shutdown.join().unwrap();
            if failed {
                assert_eq!(result.unwrap_err().to_string(), "serve failed");
            } else {
                result.unwrap();
            }
            assert!(!endpoint.socket.exists());
            contender.try_lock().unwrap();
        }
    }

    #[test]
    fn socket_selection_reports_length_for_override_and_exhausted_fallbacks() {
        let normal = PathBuf::from(format!("/{}", "long".repeat(40)));
        let limit = socket_path_limit();
        let error = select_socket_path(
            &normal,
            true,
            Some(Path::new("/temp")),
            Path::new("/tmp"),
            42,
            limit,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(&normal.display().to_string()));
        assert!(error.contains(&format!("{} bytes", normal.as_os_str().as_bytes().len())));
        assert!(error.contains(&format!("fewer than {limit} bytes")));
        assert!(error.contains("SEYK_DAEMON_DIR"));
        let error = select_socket_path(&normal, false, Some(&normal), &normal, 42, limit)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("{}/seyk-42/d-", normal.display())));
        assert!(error.contains("SEYK_DAEMON_DIR"));
    }

    #[test]
    fn socket_limit_counts_bytes_and_reserves_the_terminator() {
        let limit = socket_path_limit();
        let fits = PathBuf::from("x".repeat(limit - 1));
        assert!(check_socket_path(&fits, limit).is_ok());
        assert!(check_socket_path(&fits.with_file_name("x".repeat(limit)), limit).is_err());
        let unicode = PathBuf::from("é".repeat(limit / 2));
        assert!(check_socket_path(&unicode, limit).is_err());
        let bytes = vec![0xff; limit];
        let invalid = std::ffi::OsStr::from_bytes(&bytes);
        assert!(check_socket_path(Path::new(invalid), limit).is_err());
    }

    #[test]
    fn private_socket_directory_is_created_and_reused_with_mode_0700() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("private");
        ensure_socket_dir(&dir, current_uid()).unwrap();
        ensure_socket_dir(&dir, current_uid()).unwrap();
        let metadata = std::fs::symlink_metadata(&dir).unwrap();
        assert!(metadata.is_dir());
        assert_eq!(metadata.mode() & 0o7777, 0o700);
        assert_eq!(metadata.uid(), current_uid());
    }

    #[test]
    fn private_socket_directory_creation_restores_owner_bits_after_umask() {
        const CHILD_DIR: &str = "SEYK_TEST_SOCKET_UMASK_DIR";
        if let Some(dir) = std::env::var_os(CHILD_DIR) {
            ensure_socket_dir(Path::new(&dir), current_uid()).unwrap();
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("private");
        // umask is process-wide. Set it in a shell after exec has closed the
        // test runner's inherited descriptors, then replace the shell with the test.
        let mut child = Command::new("/bin/sh");
        child
            .args(["-c", "umask 0277\nexec \"$@\"", "seyk-umask-test"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "daemon::tests::private_socket_directory_creation_restores_owner_bits_after_umask",
                "--nocapture",
            ])
            .env(CHILD_DIR, &dir);
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o7777, 0o700);
    }

    #[test]
    fn private_socket_directory_rejects_wrong_permissions_and_owner() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("private");
        ensure_socket_dir(&dir, current_uid()).unwrap();
        // Inject a different expected uid to exercise the ownership check without root.
        let error = ensure_socket_dir(&dir, current_uid().wrapping_add(1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("owned by uid"));
        for mode in [0o755, 0o777, 0o1700, 0o500] {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
            let error = ensure_socket_dir(&dir, current_uid())
                .unwrap_err()
                .to_string();
            assert!(error.contains("mode 0700"));
            assert!(error.contains(&dir.display().to_string()));
            assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o7777, mode);
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn private_socket_directory_rejects_files_and_symlinks_without_touching_target() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        ensure_socket_dir(&target, current_uid()).unwrap();
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let error = ensure_socket_dir(&link, current_uid())
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a symlink"));
        let file = temp.path().join("file");
        std::fs::write(&file, b"keep").unwrap();
        assert!(ensure_socket_dir(&file, current_uid()).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"keep");
        assert_eq!(std::fs::metadata(&target).unwrap().mode() & 0o7777, 0o700);
    }

    #[test]
    fn stale_socket_cleanup_rechecks_directory_and_only_removes_selected_file() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("private");
        ensure_socket_dir(&dir, current_uid()).unwrap();
        let endpoint = Endpoint {
            socket: dir.join("selected.sock"),
            private_socket_dir: true,
            lock: temp.path().join("daemon.lock"),
            log: temp.path().join("daemon.log"),
            spawn_args: vec![],
        };
        std::fs::write(&endpoint.socket, b"stale").unwrap();
        let other = dir.join("other.sock");
        std::fs::write(&other, b"keep").unwrap();
        endpoint.remove_socket().unwrap();
        assert!(!endpoint.socket.exists());
        assert_eq!(std::fs::read(&other).unwrap(), b"keep");
        let moved = temp.path().join("moved");
        std::fs::rename(&dir, &moved).unwrap();
        std::fs::write(moved.join("selected.sock"), b"keep").unwrap();
        std::os::unix::fs::symlink(&moved, &dir).unwrap();
        assert!(endpoint.remove_socket().is_err());
        assert_eq!(std::fs::read(moved.join("selected.sock")).unwrap(), b"keep");
    }

    const IDLE: Duration = Duration::from_secs(60);

    fn test_hub() -> Arc<Hub> {
        Hub::new(Box::new(|| bail!("test must not load a model")), false)
    }

    async fn send_hello(stream: &mut UnixStream, hello: Hello) {
        let mut line = serde_json::to_string(&hello).unwrap();
        line.push('\n');
        stream.write_all(line.as_bytes()).await.unwrap();
    }

    #[test]
    fn session_between_ticks_resets_idle_deadline() {
        let start = Instant::now() - IDLE;
        let sessions = Arc::new(Sessions::new(start));
        assert!(!sessions.close_if_idle(start + IDLE - Duration::from_secs(5), IDLE));
        // No idle tick observes this short session.
        drop(sessions.acquire().unwrap());
        assert!(!sessions.close_if_idle(start + IDLE, IDLE));
        let released = sessions.state.lock().unwrap().last_release;
        assert!(!sessions.close_if_idle(released + IDLE - Duration::from_nanos(1), IDLE));
        assert!(sessions.close_if_idle(released + IDLE, IDLE));
        assert!(sessions.acquire().is_err());
    }

    #[test]
    fn only_last_drop_records_idle_start() {
        let start = Instant::now() - IDLE;
        let sessions = Arc::new(Sessions::new(start));
        let first = sessions.acquire().unwrap();
        let last = sessions.acquire().unwrap();
        drop(first);
        assert_eq!(sessions.active(), 1);
        assert_eq!(sessions.state.lock().unwrap().last_release, start);
        assert!(!sessions.close_if_idle(start + IDLE * 2, IDLE));
        let before = Instant::now();
        drop(last);
        let after = Instant::now();
        assert_eq!(sessions.active(), 0);
        let released = sessions.state.lock().unwrap().last_release;
        assert!((before..=after).contains(&released));
        assert!(!sessions.close_if_idle(released, IDLE));
    }

    #[test]
    fn admission_and_closing_have_one_winner() {
        for _ in 0..32 {
            let now = Instant::now();
            let sessions = Arc::new(Sessions::new(now - IDLE));
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let admission = std::thread::spawn({
                let sessions = sessions.clone();
                let barrier = barrier.clone();
                move || {
                    barrier.wait();
                    sessions.acquire()
                }
            });
            barrier.wait();
            let closed = sessions.close_if_idle(now, IDLE);
            // The thread returns its lease, keeping it alive until both racers finish.
            let lease = admission.join().unwrap();
            assert_eq!(closed, lease.is_err());
            assert_eq!(sessions.active(), usize::from(lease.is_ok()));
            if closed {
                assert!(sessions.acquire().is_err());
            } else {
                assert!(!sessions.close_if_idle(now + IDLE, IDLE));
            }
        }
    }

    #[test]
    fn setup_crossing_idle_deadline_holds_a_lease() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        rt.block_on(async {
            // Occupy the blocking pool so the real handler cannot construct its
            // server yet. Admission must already protect that queued setup.
            let (release, wait) = std::sync::mpsc::channel();
            let (started, ready) = tokio::sync::oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                let _ = wait.recv();
            });
            ready.await.unwrap();
            let now = Instant::now();
            let sessions = Arc::new(Sessions::new(now - IDLE));
            let (mut client, server) = UnixStream::pair().unwrap();
            send_hello(&mut client, Hello::Session { roots: vec![] }).await;
            let (stop, _) = tokio::sync::mpsc::channel(1);
            let task = tokio::spawn(handle(server, test_hub(), sessions.clone(), stop));
            let admitted = tokio::time::timeout(Duration::from_secs(5), async {
                while sessions.active() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await;
            let closed = sessions.close_if_idle(now + IDLE, IDLE);
            // Always unblock the pool before assertions (runtime drop waits for it).
            release.send(()).unwrap();
            blocker.await.unwrap();
            task.await.unwrap().unwrap();
            assert!(admitted.is_ok());
            assert!(!closed);
            assert_eq!(sessions.active(), 0);
        });
    }

    #[tokio::test]
    async fn failed_setup_and_ack_release_the_lease() {
        let dir = tempfile::tempdir().unwrap();
        for roots in [vec![], vec![dir.path().to_path_buf()]] {
            let start = Instant::now() - IDLE;
            let sessions = Arc::new(Sessions::new(start));
            let (mut client, server) = UnixStream::pair().unwrap();
            send_hello(&mut client, Hello::Session { roots }).await;
            // An empty root list fails setup; a valid root reaches the failed ack.
            drop(client);
            let (stop, _) = tokio::sync::mpsc::channel(1);
            assert!(
                handle(server, test_hub(), sessions.clone(), stop)
                    .await
                    .is_err()
            );
            assert_eq!(sessions.active(), 0);
            assert!(sessions.state.lock().unwrap().last_release > start);
            assert!(!sessions.close_if_idle(Instant::now(), IDLE));
        }
    }

    #[tokio::test]
    async fn cancelled_serving_future_releases_the_lease() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = Arc::new(Sessions::new(Instant::now() - IDLE));
        let (mut client, server) = UnixStream::pair().unwrap();
        send_hello(
            &mut client,
            Hello::Session {
                roots: vec![dir.path().into()],
            },
        )
        .await;
        let (stop, _) = tokio::sync::mpsc::channel(1);
        let task = tokio::spawn(handle(server, test_hub(), sessions.clone(), stop));
        let mut answer = String::new();
        BufReader::new(client).read_line(&mut answer).await.unwrap();
        assert!(serde_json::from_str::<Reply>(&answer).unwrap().ok);
        assert_eq!(sessions.active(), 1);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(sessions.active(), 0);
        assert!(!sessions.close_if_idle(Instant::now(), IDLE));
    }

    #[tokio::test]
    async fn panic_in_serving_future_releases_the_lease() {
        let sessions = Arc::new(Sessions::new(Instant::now() - IDLE));
        let task = tokio::spawn({
            let sessions = sessions.clone();
            async move {
                let _lease = sessions.acquire().unwrap();
                tokio::task::yield_now().await;
                panic!("serving future failed");
            }
        });
        assert!(task.await.unwrap_err().is_panic());
        assert_eq!(sessions.active(), 0);
        assert!(!sessions.close_if_idle(Instant::now(), IDLE));
    }

    #[tokio::test]
    async fn closing_refuses_sessions_with_an_error_reply() {
        let now = Instant::now();
        let sessions = Arc::new(Sessions::new(now - IDLE));
        assert!(sessions.close_if_idle(now, IDLE));
        let (mut client, server) = UnixStream::pair().unwrap();
        send_hello(&mut client, Hello::Session { roots: vec![] }).await;
        let (stop, _) = tokio::sync::mpsc::channel(1);
        handle(server, test_hub(), sessions.clone(), stop)
            .await
            .unwrap();
        let mut answer = String::new();
        BufReader::new(client).read_line(&mut answer).await.unwrap();
        let reply: Reply = serde_json::from_str(&answer).unwrap();
        assert!(!reply.ok);
        assert_eq!(reply.error.as_deref(), Some("daemon is closing"));
        assert_eq!(sessions.active(), 0);
    }

    #[tokio::test]
    async fn status_and_stop_do_not_reset_idle() {
        let now = Instant::now();
        let sessions = Arc::new(Sessions::new(now - IDLE));
        for hello in [Hello::Status, Hello::Stop] {
            let (mut client, server) = UnixStream::pair().unwrap();
            send_hello(&mut client, hello).await;
            let (stop, mut stopped) = tokio::sync::mpsc::channel(1);
            handle(server, test_hub(), sessions.clone(), stop)
                .await
                .unwrap();
            let mut answer = String::new();
            BufReader::new(client).read_line(&mut answer).await.unwrap();
            let reply: Reply = serde_json::from_str(&answer).unwrap();
            assert!(reply.ok);
            if let Some(status) = reply.status {
                assert_eq!(status.sessions, 0);
                assert_eq!(status.version, env!("SEYK_BUILD_VERSION"));
            } else {
                assert_eq!(stopped.recv().await, Some(()));
            }
            assert_eq!(sessions.active(), 0);
            assert_eq!(sessions.state.lock().unwrap().last_release, now - IDLE);
        }
        assert!(sessions.close_if_idle(now, IDLE));
    }

    #[test]
    fn endpoint_separates_build_cache_protocol_and_model_arguments() {
        let args = vec!["--dimension".into(), "256".into()];
        let id = endpoint_id(&args, "0.1.0+g123", 9, "7");
        assert_eq!(id, endpoint_id(&args, "0.1.0+g123", 9, "7"));
        assert_ne!(id, endpoint_id(&args, "0.1.0+g456", 9, "7"));
        assert_ne!(id, endpoint_id(&args, "0.1.0+g123.dirty", 9, "7"));
        assert_ne!(id, endpoint_id(&args, "0.1.0+g123", 9, "8"));
        assert_ne!(id, endpoint_id(&args, "0.1.0+g123", 10, "7"));
        assert_ne!(
            id,
            endpoint_id(&["--dimension".into(), "128".into()], "0.1.0+g123", 9, "7")
        );
        assert_ne!(
            endpoint_id(&["ab".into(), "c".into()], "v", 9, "7"),
            endpoint_id(&["a".into(), "bc".into()], "v", 9, "7")
        );
        // An already-running pre-build-identity daemon keeps its own endpoint.
        let mut legacy = blake3::Hasher::new();
        legacy.update(env!("CARGO_PKG_VERSION").as_bytes());
        legacy.update(b"json-schema");
        legacy.update(&crate::JSON_SCHEMA_VERSION.to_le_bytes());
        for a in &args {
            legacy.update(&(a.len() as u64).to_le_bytes());
            legacy.update(a.as_bytes());
        }
        assert_ne!(
            endpoint_id(
                &args,
                env!("SEYK_BUILD_VERSION"),
                crate::JSON_SCHEMA_VERSION,
                seyk::cache::SCHEMA_VERSION
            ),
            &legacy.finalize().to_hex()[..12]
        );
    }
}
