//! One background process per configuration that every agent session shares.
//!
//! `seyk mcp` is a thin relay: it connects to the daemon's Unix socket (starting
//! the daemon if none is running), sends one hello line naming its folders, then
//! copies MCP bytes between the agent's stdio and the socket. The daemon holds the
//! only copy of the model and one watcher per folder, so five agent sessions cost
//! one model's memory instead of five. It exits after a quiet period with no
//! sessions, so the memory is only held while agents use it.
//!
//! The socket name includes a hash of the Seyk version and model options. A
//! session never reaches a daemon running other weights or an older binary; an
//! old daemon simply idles out after an upgrade.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
}

// ---- daemon side -----------------------------------------------------------

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

    let sessions = Arc::new(AtomicUsize::new(0));
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    let mut quiet_since = Some(Instant::now());
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            conn = listener.accept() => {
                let (stream, _) = conn?;
                let (hub, sessions, stop) = (hub.clone(), sessions.clone(), stop_tx.clone());
                tokio::spawn(async move {
                    if let Err(e) = handle(stream, hub, sessions, stop).await {
                        eprintln!("seyk daemon: connection: {e:#}");
                    }
                });
            }
            _ = stop_rx.recv() => {
                eprintln!("seyk daemon: stopped on request");
                return Ok(());
            }
            _ = tick.tick() => {
                if sessions.load(Ordering::SeqCst) > 0 {
                    quiet_since = None;
                } else {
                    let since = *quiet_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= idle {
                        eprintln!("seyk daemon: idle for {}s, exiting", idle.as_secs());
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn handle(
    stream: UnixStream,
    hub: Arc<Hub>,
    sessions: Arc<AtomicUsize>,
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
                sessions: sessions.load(Ordering::SeqCst),
                folders: hub.watched(),
            };
            reply(&mut write, Ok(Some(status))).await
        }
        Hello::Stop => {
            reply(&mut write, Ok(None)).await?;
            let _ = stop.send(()).await;
            Ok(())
        }
        Hello::Session { roots } => {
            // Building a session canonicalizes roots and may start watchers.
            let server =
                match tokio::task::spawn_blocking(move || SeykServer::new(roots, hub)).await? {
                    Ok(s) => s,
                    Err(e) => return reply(&mut write, Err(e)).await,
                };
            reply(&mut write, Ok(None)).await?;
            sessions.fetch_add(1, Ordering::SeqCst);
            // The buffered reader keeps any MCP bytes that arrived with the hello.
            let result = async {
                let service = server.serve((read, write)).await?;
                service.waiting().await?;
                anyhow::Ok(())
            }
            .await;
            sessions.fetch_sub(1, Ordering::SeqCst);
            result
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
    // Canonical here, so the daemon sees the same folders whatever its cwd.
    let roots = roots
        .iter()
        .map(|r| {
            r.canonicalize()
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
