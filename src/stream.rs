//! The root daemon's sample stream: a local Unix socket, no network.
//!
//! The daemon listens on `stream_socket` (mode 0666 in a 0755 directory: anyone on
//! the host may read the samples, which are not secret). Each connected client gets
//! every sample as one line of JSON. The daemon never reads from a client and never
//! blocks on one: a client that cannot take a whole line at once is dropped.
//! `gpu-powerd-ws` (unprivileged, its own unit) is the client that serves WebSocket.
use std::io::{ErrorKind, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

/// At most this many clients at once; more are refused.
const MAX_CLIENTS: usize = 16;

pub struct Publisher {
    listener: UnixListener,
    path: PathBuf,
    clients: Vec<UnixStream>,
}

impl Publisher {
    pub fn bind(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        // A stale socket from a previous run (the single-instance lock is held).
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != ErrorKind::NotFound => return Err(e),
            _ => {}
        }
        let listener = UnixListener::bind(path)?;
        listener.set_nonblocking(true)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666))?;
        Ok(Self { listener, path: path.to_path_buf(), clients: Vec::new() })
    }

    /// Accept waiting clients, then send `line` to each; drop any that would block.
    pub fn publish(&mut self, line: &[u8]) {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if self.clients.len() >= MAX_CLIENTS || stream.set_nonblocking(true).is_err() {
                        continue;
                    }
                    let _ = stream.shutdown(std::net::Shutdown::Read);
                    self.clients.push(stream);
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
        self.clients.retain_mut(|client| matches!(client.write(line), Ok(n) if n == line.len()));
    }

    pub fn clients(&self) -> usize {
        self.clients.len()
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    #[test]
    fn clients_get_every_line_and_stalled_ones_are_dropped() {
        let dir = std::env::temp_dir().join(format!("gpu-powerd-stream-{}", std::process::id()));
        let path = dir.join("stream.sock");
        let mut publisher = Publisher::bind(&path).unwrap();
        let reader = UnixStream::connect(&path).unwrap();
        let stalled = UnixStream::connect(&path).unwrap();
        publisher.publish(b"{\"a\":1}\n");
        assert_eq!(publisher.clients(), 2);
        let mut lines = BufReader::new(reader);
        let mut line = String::new();
        lines.read_line(&mut line).unwrap();
        assert_eq!(line, "{\"a\":1}\n");
        // The stalled client never reads: its buffer fills and it is dropped, without
        // the publisher ever blocking.
        let big = vec![b'x'; 64 * 1024];
        for _ in 0..64 {
            publisher.publish(&big);
            let mut sink = vec![0u8; 64 * 1024];
            let _ = std::io::Read::read(lines.get_mut(), &mut sink);
        }
        assert!(publisher.clients() <= 1, "the stalled client was dropped");
        drop(stalled);
        drop(publisher);
        assert!(!path.exists(), "the socket is removed on exit");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
