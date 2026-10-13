//! A minimal, send-only WebSocket server side (RFC 6455): the opening handshake and
//! unmasked text frames. Client frames are read only to notice a close or answer a
//! ping; their content is ignored. No extensions, no fragmentation, no TLS.
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const MAX_HEADER_BYTES: usize = 8 * 1024;

/// SHA-1 (FIPS 180-4), for the handshake's accept key only.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut message = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());
    for chunk in message.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

pub fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn accept_key(client_key: &str) -> String {
    base64(&sha1(format!("{}{GUID}", client_key.trim()).as_bytes()))
}

/// Read the HTTP upgrade request and answer it. Anything else gets a short plain-text
/// reply (so a browser pointed at the port sees what it is) and an error.
pub fn handshake(stream: &mut TcpStream) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut key = None;
    let mut upgrade = false;
    let mut total = 0;
    let mut first = true;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        total += n;
        if n == 0 || total > MAX_HEADER_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request"));
        }
        let line = line.trim_end();
        if first {
            first = false;
            if !line.starts_with("GET ") {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GET"));
            }
            continue;
        }
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            if name == "sec-websocket-key" {
                key = Some(value.trim().to_string());
            } else if name == "upgrade" && value.trim().eq_ignore_ascii_case("websocket") {
                upgrade = true;
            }
        }
    }
    let Some(key) = key.filter(|_| upgrade) else {
        let body = "gpu-powerd-ws: connect with a WebSocket client for the GPU sample stream\n";
        let _ = write!(
            stream,
            "HTTP/1.1 426 Upgrade Required\r\nUpgrade: websocket\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a WebSocket upgrade"));
    };
    write!(
        stream,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        accept_key(&key)
    )?;
    stream.flush()
}

/// An unmasked server frame.
pub fn frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    out.push(0x80 | opcode);
    match payload.len() {
        n if n < 126 => out.push(n as u8),
        n if n <= 0xFFFF => {
            out.push(126);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            out.push(127);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    out.extend_from_slice(payload);
    out
}

pub fn text(payload: &[u8]) -> Vec<u8> {
    frame(0x1, payload)
}

/// What a client sent: only control frames matter.
pub enum Incoming {
    Close,
    Ping(Vec<u8>),
    Ignored,
}

/// Read one client frame (masked, per RFC 6455), discarding data payloads. Frames
/// larger than 64 KiB end the connection.
pub fn read_frame(reader: &mut impl Read) -> io::Result<Incoming> {
    let mut head = [0u8; 2];
    reader.read_exact(&mut head)?;
    let opcode = head[0] & 0x0F;
    let masked = head[1] & 0x80 != 0;
    let mut len = u64::from(head[1] & 0x7F);
    if len == 126 {
        let mut b = [0u8; 2];
        reader.read_exact(&mut b)?;
        len = u64::from(u16::from_be_bytes(b));
    } else if len == 127 {
        let mut b = [0u8; 8];
        reader.read_exact(&mut b)?;
        len = u64::from_be_bytes(b);
    }
    if len > 64 * 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "client frame too large"));
    }
    let mut mask = [0u8; 4];
    if masked {
        reader.read_exact(&mut mask)?;
    }
    let mut payload = vec![0u8; len as usize];
    reader.read_exact(&mut payload)?;
    for (i, b) in payload.iter_mut().enumerate() {
        *b ^= mask[i % 4];
    }
    Ok(match opcode {
        0x8 => Incoming::Close,
        0x9 => Incoming::Ping(payload),
        _ => Incoming::Ignored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_base64_and_rfc_accept_key() {
        let hex: String = sha1(b"abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(base64(b"hello"), "aGVsbG8=");
        assert_eq!(base64(b"hi!"), "aGkh");
        // RFC 6455 section 1.3.
        assert_eq!(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn frames() {
        assert_eq!(text(b"hi"), vec![0x81, 2, b'h', b'i']);
        let long = text(&[0u8; 300]);
        assert_eq!(&long[..4], &[0x81, 126, 1, 44]);
        let masked = [0x89u8, 0x82, 1, 2, 3, 4, b'o' ^ 1, b'k' ^ 2];
        match read_frame(&mut &masked[..]).unwrap() {
            Incoming::Ping(p) => assert_eq!(p, b"ok"),
            _ => panic!("expected a ping"),
        }
    }
}
