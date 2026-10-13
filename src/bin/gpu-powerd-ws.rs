//! `gpu-powerd-ws`: relay gpu-powerd's sample stream to WebSocket clients.
//!
//! Runs unprivileged (its unit uses `DynamicUser=yes` with no capabilities). It
//! connects to the root daemon's local stream socket and pushes every sample, one JSON
//! text frame each, to every connected WebSocket client. It is read-only: client
//! messages are ignored (only close and ping are honoured) and there are no commands.
//! There is no authentication; bind it to a trusted network.
use anyhow::{Context, Result, bail};
use gpu_powerd::config::{Config, DEFAULT_CONFIG};
use gpu_powerd::websocket::{self, Incoming};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Frames buffered per client; a client further behind is dropped.
const CLIENT_QUEUE: usize = 32;
const MAX_CLIENTS: usize = 64;

type Clients = Arc<Mutex<Vec<SyncSender<Arc<Vec<u8>>>>>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("gpu-powerd-ws: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut config_path = PathBuf::from(DEFAULT_CONFIG);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => config_path = args.next().context("--config needs a path")?.into(),
            "-V" | "--version" => {
                println!("gpu-powerd-ws {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "-h" | "--help" => {
                println!("gpu-powerd-ws [--config PATH]: relay gpu-powerd samples to WebSocket clients (read-only)");
                return Ok(());
            }
            _ => bail!("unexpected argument {arg:?}"),
        }
    }
    let config = if config_path.exists() { Config::load(&config_path)? } else { Config::default() };
    if !config.websocket {
        eprintln!("gpu-powerd-ws: websocket = false in {}; exiting", config_path.display());
        return Ok(());
    }
    let address = format!("{}:{}", config.websocket_bind, config.websocket_port);
    let address = if config.websocket_bind.contains(':') {
        format!("[{}]:{}", config.websocket_bind, config.websocket_port)
    } else {
        address
    };
    let listener = TcpListener::bind(&address).with_context(|| format!("bind {address}"))?;
    eprintln!(
        "gpu-powerd-ws: serving {} on ws://{address}/ (read-only, no authentication)",
        config.stream_socket.display()
    );
    let clients: Clients = Arc::new(Mutex::new(Vec::new()));
    let relay_clients = clients.clone();
    let socket = config.stream_socket.clone();
    std::thread::spawn(move || relay(&socket, &relay_clients));
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let clients = clients.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve(stream, &clients) {
                eprintln!("gpu-powerd-ws: client: {e}");
            }
        });
    }
    Ok(())
}

/// Read samples from the daemon (reconnecting while it is down) and fan them out.
fn relay(socket: &std::path::Path, clients: &Clients) {
    let mut warned = false;
    loop {
        match UnixStream::connect(socket) {
            Ok(stream) => {
                warned = false;
                eprintln!("gpu-powerd-ws: connected to {}", socket.display());
                let mut reader = BufReader::new(stream);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match reader.read_until(b'\n', &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let payload = line.strip_suffix(b"\n").unwrap_or(&line);
                            let frame = Arc::new(websocket::text(payload));
                            let mut list = clients.lock().unwrap_or_else(|e| e.into_inner());
                            list.retain(|tx| {
                                !matches!(
                                    tx.try_send(frame.clone()),
                                    Err(TrySendError::Disconnected(_) | TrySendError::Full(_))
                                )
                            });
                        }
                    }
                }
                eprintln!("gpu-powerd-ws: the daemon closed the stream; reconnecting");
            }
            Err(e) if !warned => {
                warned = true;
                eprintln!("gpu-powerd-ws: waiting for {} ({e})", socket.display());
            }
            Err(_) => {}
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn serve(mut stream: TcpStream, clients: &Clients) -> std::io::Result<()> {
    {
        let list = clients.lock().unwrap_or_else(|e| e.into_inner());
        if list.len() >= MAX_CLIENTS {
            return Err(std::io::Error::other("too many clients"));
        }
    }
    websocket::handshake(&mut stream)?;
    stream.set_read_timeout(None)?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    stream.set_nodelay(true)?;
    let (tx, rx) = mpsc::sync_channel::<Arc<Vec<u8>>>(CLIENT_QUEUE);
    // Control replies (pong, close) go through the same writer.
    let control = tx.clone();
    clients.lock().unwrap_or_else(|e| e.into_inner()).push(tx);
    let mut reader = stream.try_clone()?;
    std::thread::spawn(move || {
        loop {
            match websocket::read_frame(&mut reader) {
                Ok(Incoming::Ping(payload)) => {
                    let _ = control.try_send(Arc::new(websocket::frame(0xA, &payload)));
                }
                Ok(Incoming::Ignored) => {}
                Ok(Incoming::Close) | Err(_) => {
                    let _ = control.try_send(Arc::new(websocket::frame(0x8, &[])));
                    let _ = reader.shutdown(std::net::Shutdown::Read);
                    break;
                }
            }
        }
    });
    for frame in rx {
        stream.write_all(&frame)?;
        if frame.first() == Some(&0x88) {
            break;
        }
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}
