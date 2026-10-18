//! Bounded MPSC mailbox fabric. Send never takes a process-wide HashMap lock.

use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock, RwLock};

pub const MAILBOX_BOUND: usize = 128;
pub const ACTOR_STACK: usize = 256 * 1024;
const CHUNK: usize = 4096;

pub fn actor_stack_size() -> usize {
    ACTOR_STACK
}

pub fn channel<T: Send>() -> (SyncSender<T>, Receiver<T>) {
    mpsc::sync_channel(MAILBOX_BOUND)
}

struct Chunk<T: Send> {
    slots: Box<[Mutex<Option<SyncSender<T>>>]>,
}

impl<T: Send> Chunk<T> {
    fn new() -> Self {
        let mut slots = Vec::with_capacity(CHUNK);
        for _ in 0..CHUNK {
            slots.push(Mutex::new(None));
        }
        Chunk {
            slots: slots.into_boxed_slice(),
        }
    }
}

pub struct MailboxTable<T: Send> {
    chunks: RwLock<Vec<OnceLock<Chunk<T>>>>,
}

impl<T: Send> MailboxTable<T> {
    pub fn new() -> Self {
        MailboxTable {
            chunks: RwLock::new(Vec::new()),
        }
    }

    fn chunk(chunks: &[OnceLock<Chunk<T>>], id: usize) -> Option<&Chunk<T>> {
        chunks.get(id / CHUNK).and_then(|c| c.get())
    }

    fn ensure_chunk(&self, id: usize) -> usize {
        let idx = id / CHUNK;
        {
            let g = self.chunks.read().unwrap();
            if idx < g.len() && g[idx].get().is_some() {
                return idx;
            }
        }
        let mut g = self.chunks.write().unwrap();
        if idx >= g.len() {
            g.resize_with(idx + 1, OnceLock::new);
        }
        let _ = g[idx].set(Chunk::new());
        idx
    }

    pub fn register(&self, id: usize, tx: SyncSender<T>) {
        let idx = self.ensure_chunk(id);
        let g = self.chunks.read().unwrap();
        let chunk = g[idx].get().expect("chunk inserted");
        *chunk.slots[id % CHUNK].lock().unwrap() = Some(tx);
    }

    pub fn send(&self, id: usize, msg: T) {
        let g = self.chunks.read().unwrap();
        let Some(chunk) = Self::chunk(&g, id) else {
            return;
        };
        let tx = {
            let slot = chunk.slots[id % CHUNK].lock().unwrap();
            slot.clone()
        };
        drop(g);
        if let Some(tx) = tx {
            match tx.try_send(msg) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => {}
                Err(TrySendError::Full(msg)) => {
                    let _ = msg;
                }
            }
        }
    }

    pub fn send_blocking(&self, id: usize, msg: T) {
        let g = self.chunks.read().unwrap();
        let Some(chunk) = Self::chunk(&g, id) else {
            return;
        };
        let tx = {
            let slot = chunk.slots[id % CHUNK].lock().unwrap();
            slot.clone()
        };
        drop(g);
        if let Some(tx) = tx {
            let _ = tx.send(msg);
        }
    }

    pub fn broadcast(&self, make: impl Fn() -> T)
    where
        T: Clone,
    {
        let senders: Vec<SyncSender<T>> = {
            let g = self.chunks.read().unwrap();
            let mut out = Vec::new();
            for chunk in g.iter().filter_map(|c| c.get()) {
                for slot in chunk.slots.iter() {
                    if let Some(tx) = slot.lock().unwrap().clone() {
                        out.push(tx);
                    }
                }
            }
            out
        };
        for tx in senders {
            let _ = tx.send(make());
        }
    }

    pub fn for_each_sender(&self, mut f: impl FnMut(&SyncSender<T>)) {
        let g = self.chunks.read().unwrap();
        for chunk in g.iter().filter_map(|c| c.get()) {
            for slot in chunk.slots.iter() {
                if let Some(tx) = slot.lock().unwrap().clone() {
                    f(&tx);
                }
            }
        }
    }
}

impl<T: Send> Default for MailboxTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_send_does_not_block() {
        let table = MailboxTable::<i64>::new();
        let (tx, rx) = channel();
        table.register(0, tx);
        for i in 0..(MAILBOX_BOUND as i64 + 32) {
            table.send(0, i);
        }
        let mut n = 0;
        while rx.try_recv().is_ok() {
            n += 1;
        }
        assert_eq!(n, MAILBOX_BOUND);
    }

    #[test]
    fn unknown_id_is_noop() {
        let table = MailboxTable::<i64>::new();
        table.send(99, 1);
    }
}
