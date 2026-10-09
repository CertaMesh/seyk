//! Every byte served is checked against the same pass that produced the source hash.
use crate::{Error, ResourceId, Result, provider::ReadSeek};
use std::{
    collections::VecDeque,
    io::{self, Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
const CHUNK: usize = 1024 * 1024;
const CACHE_CHUNKS: usize = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct VerificationStats {
    pub hash_bytes: u64,
    pub hash_time: Duration,
    pub verified_bytes: u64,
    pub verify_time: Duration,
}

/// An operation-local chunk commitment. No source data or hashes are persisted.
pub struct VerifiedReader {
    source: Box<dyn ReadSeek>,
    hashes: Vec<blake3::Hash>,
    content_hash: String,
    len: u64,
    pos: u64,
    cache: VecDeque<(u64, Vec<u8>)>,
    changed: Arc<AtomicBool>,
    stats: VerificationStats,
}
impl VerifiedReader {
    pub fn new(mut source: Box<dyn ReadSeek>, id: &ResourceId, max_bytes: u64) -> Result<Self> {
        let started = Instant::now();
        let fail = |_| Error::UnsupportedAudio {
            resource: id.clone(),
            reason: "cannot read audio source",
        };
        let len = source.seek(SeekFrom::End(0)).map_err(fail)?;
        if len > max_bytes {
            return Err(Error::UnsupportedAudio {
                resource: id.clone(),
                reason: "audio exceeds duration-derived byte limit",
            });
        }
        source.seek(SeekFrom::Start(0)).map_err(fail)?;
        let mut hash = blake3::Hasher::new();
        let mut hashes = Vec::new();
        let mut buf = vec![0; CHUNK];
        let mut remaining = len;
        while remaining > 0 {
            let n = remaining.min(CHUNK as u64) as usize;
            source.read_exact(&mut buf[..n]).map_err(fail)?;
            hash.update(&buf[..n]);
            hashes.push(blake3::hash(&buf[..n]));
            remaining -= n as u64;
        }
        if source.seek(SeekFrom::End(0)).map_err(fail)? != len {
            return Err(Error::ChangedWhileReading {
                resource: id.clone(),
            });
        }
        Ok(Self {
            source,
            hashes,
            content_hash: hash.finalize().to_hex().to_string(),
            len,
            pos: 0,
            cache: VecDeque::new(),
            changed: Arc::new(AtomicBool::new(false)),
            stats: VerificationStats {
                hash_bytes: len,
                hash_time: started.elapsed(),
                ..Default::default()
            },
        })
    }
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn stats(&self) -> VerificationStats {
        self.stats
    }
    pub(super) fn changed_flag(&self) -> Arc<AtomicBool> {
        self.changed.clone()
    }
    fn changed(&self) -> io::Error {
        self.changed.store(true, Ordering::Relaxed);
        io::Error::new(io::ErrorKind::InvalidData, "audio changed while reading")
    }
}
impl Read for VerifiedReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.changed.load(Ordering::Relaxed) {
            return Err(self.changed());
        }
        if out.is_empty() || self.pos >= self.len {
            return Ok(0);
        }
        let chunk = self.pos / CHUNK as u64;
        if !self.cache.iter().any(|(i, _)| *i == chunk) {
            let started = Instant::now();
            let offset = chunk * CHUNK as u64;
            let n = (self.len - offset).min(CHUNK as u64) as usize;
            let mut buf = vec![0; n];
            let read = (|| {
                if self.source.seek(SeekFrom::End(0))? != self.len {
                    return Err(io::Error::other("length changed"));
                }
                self.source.seek(SeekFrom::Start(offset))?;
                self.source.read_exact(&mut buf)
            })();
            if read.is_err() || blake3::hash(&buf) != self.hashes[chunk as usize] {
                return Err(self.changed());
            }
            self.stats.verified_bytes += n as u64;
            self.stats.verify_time += started.elapsed();
            if self.cache.len() == CACHE_CHUNKS {
                self.cache.pop_front();
            }
            self.cache.push_back((chunk, buf));
        }
        let buf = &self.cache.iter().find(|(i, _)| *i == chunk).unwrap().1;
        let offset = (self.pos % CHUNK as u64) as usize;
        let n = out.len().min(buf.len() - offset);
        out[..n].copy_from_slice(&buf[offset..offset + n]);
        self.pos += n as u64;
        Ok(n)
    }
}
impl Seek for VerifiedReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let n = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(n) => self.pos as i128 + n as i128,
            SeekFrom::End(n) => self.len as i128 + n as i128,
        };
        self.pos = u64::try_from(n)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid seek"))?;
        Ok(self.pos)
    }
}
impl symphonia::core::io::MediaSource for VerifiedReader {
    fn is_seekable(&self) -> bool {
        true
    }
    fn byte_len(&self) -> Option<u64> {
        Some(self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn changed_chunks_never_reach_callers_including_after_seek() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![7; CHUNK * 4 + 13]).unwrap();
        let id: ResourceId = "mem://verified/test".parse().unwrap();
        let mut r = VerifiedReader::new(Box::new(file.reopen().unwrap()), &id, u64::MAX).unwrap();
        let expected = blake3::hash(&vec![7; CHUNK * 4 + 13]).to_hex().to_string();
        assert_eq!(r.content_hash(), expected);
        let mut b = [0; 16];
        r.read_exact(&mut b).unwrap();
        assert_eq!(b, [7; 16]);
        // Evict the first chunk, mutate it, then seek back. Cache hits may serve
        // original verified bytes; a refetch must verify the new bytes again.
        for n in [2, 3] {
            r.seek(SeekFrom::Start((n * CHUNK) as u64)).unwrap();
            r.read_exact(&mut b).unwrap();
        }
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&[9]).unwrap();
        file.flush().unwrap();
        r.rewind().unwrap();
        b.fill(42);
        assert!(r.read(&mut b).is_err());
        assert_eq!(b, [42; 16]);
        assert!(r.changed_flag().load(Ordering::Relaxed));
    }
    #[test]
    fn mutation_after_commit_and_truncation_fail_closed() {
        for truncate in [false, true] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&vec![7; CHUNK + 13]).unwrap();
            let id: ResourceId = "mem://verified/test".parse().unwrap();
            let mut r =
                VerifiedReader::new(Box::new(file.reopen().unwrap()), &id, u64::MAX).unwrap();
            if truncate {
                file.as_file().set_len(10).unwrap();
            } else {
                file.seek(SeekFrom::Start(0)).unwrap();
                file.write_all(&[9]).unwrap();
                file.flush().unwrap();
            }
            assert!(r.read(&mut [0; 1]).is_err());
        }
    }
}
