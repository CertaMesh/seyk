//! One background process per configuration that every agent session shares.
//!
//! `seyk mcp` is a thin relay: it connects to the daemon's Unix socket (starting
//! the daemon if none is running), sends one hello line naming its folders, then
//! copies MCP bytes between the agent's stdio and the socket. The daemon holds the
//! only copy of the model and one watcher per folder, so five agent sessions cost
//! one model's memory instead of five. It exits after a quiet period with no
//! sessions, so the memory is only held while agents use it.
//!
//! The socket name hashes the package version, JSON schema version and model
//! options. Builds with the same values share an endpoint, even when their
//! implementation differs.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
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
    lock: PathBuf,
    log: PathBuf,
    /// Arguments that start a daemon with this configuration.
    spawn_args: Vec<String>,
}

impl Endpoint {
    /// `model_args` are the model flags, exactly as they would be passed to
    /// `seyk daemon run`. They and the version decide which daemon to use.
    pub fn new(model_args: Vec<String>) -> Result<Self> {
        let dir = match std::env::var_os("SEYK_DAEMON_DIR") {
            Some(d) => PathBuf::from(d),
            None => dirs::cache_dir()
                .ok_or_else(|| anyhow!("no cache directory on this platform"))?
                .join("seyk"),
        };
        std::fs::create_dir_all(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let id = endpoint_id(&model_args, crate::JSON_SCHEMA_VERSION);
        let mut spawn_args = vec!["daemon".to_string(), "run".to_string()];
        spawn_args.extend(model_args);
        Ok(Self {
            socket: dir.join(format!("daemon-{id}.sock")),
            lock: dir.join(format!("daemon-{id}.lock")),
            log: dir.join("daemon.log"),
            spawn_args,
        })
    }
}

fn endpoint_id(model_args: &[String], schema_version: u32) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
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
        eprintln!("seyk daemon: already running for this configuration");
        return Ok(());
    }
    // Holding the lock means any socket file left here is stale.
    let _ = std::fs::remove_file(&endpoint.socket);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = rt.block_on(serve(endpoint, Hub::new(factory, true), idle));
    let _ = std::fs::remove_file(&endpoint.socket);
    drop(lock);
    result
}

async fn serve(endpoint: &Endpoint, hub: Arc<Hub>, idle: Duration) -> Result<()> {
    let listener = UnixListener::bind(&endpoint.socket)
        .with_context(|| format!("binding {}", endpoint.socket.display()))?;
    std::fs::set_permissions(&endpoint.socket, std::fs::Permissions::from_mode(0o600))?;
    eprintln!(
        "seyk daemon {} (pid {}) listening on {}",
        env!("CARGO_PKG_VERSION"),
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
                version: env!("CARGO_PKG_VERSION").into(),
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
            seyk::discovery::absolute_logical_path(r)
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
            } else {
                assert_eq!(stopped.recv().await, Some(()));
            }
            assert_eq!(sessions.active(), 0);
            assert_eq!(sessions.state.lock().unwrap().last_release, now - IDLE);
        }
        assert!(sessions.close_if_idle(now, IDLE));
    }

    #[test]
    fn endpoint_changes_with_protocol_and_model_arguments() {
        let args = vec!["--dimension".into(), "256".into()];
        assert_eq!(endpoint_id(&args, 2), endpoint_id(&args, 2));
        assert_ne!(endpoint_id(&args, 1), endpoint_id(&args, 2));
        assert_ne!(
            endpoint_id(&args, 2),
            endpoint_id(&["--dimension".into(), "128".into()], 2)
        );
        let mut legacy = blake3::Hasher::new();
        legacy.update(env!("CARGO_PKG_VERSION").as_bytes());
        for a in &args {
            legacy.update(&(a.len() as u64).to_le_bytes());
            legacy.update(a.as_bytes());
        }
        assert_ne!(endpoint_id(&args, 2), &legacy.finalize().to_hex()[..12]);
    }
}
