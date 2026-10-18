//! M:N actor scheduler: a fixed worker pool plus dedicated mailbox threads.

use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::{self, JoinHandle};

pub type Job = Box<dyn FnOnce() + Send + 'static>;

enum Task {
    Run(Job),
    Shutdown,
}

pub struct Scheduler {
    tx: Sender<Task>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

fn worker_count() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 64)
}

impl Scheduler {
    pub fn global() -> &'static Scheduler {
        static SCHED: OnceLock<Scheduler> = OnceLock::new();
        SCHED.get_or_init(Scheduler::new)
    }

    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = std::sync::Arc::new(Mutex::new(rx));
        let mut workers = Vec::new();
        for i in 0..worker_count() {
            let rx = std::sync::Arc::clone(&rx);
            let handle = thread::Builder::new()
                .name(format!("tridentix-sched-{i}"))
                .spawn(move || loop {
                    let task = {
                        let guard = rx.lock().unwrap();
                        guard.recv()
                    };
                    match task {
                        Ok(Task::Run(job)) => job(),
                        Ok(Task::Shutdown) | Err(_) => break,
                    }
                })
                .expect("scheduler worker");
            workers.push(handle);
        }
        Scheduler {
            tx,
            workers: Mutex::new(workers),
        }
    }

    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(Task::Run(job));
    }

    pub fn spawn_blocking<F>(&self, name: &str, stack: usize, f: F) -> std::io::Result<JoinHandle<()>>
    where
        F: FnOnce() + Send + 'static,
    {
        thread::Builder::new().name(name.to_string()).stack_size(stack).spawn(f)
    }

    pub fn shutdown(&self) {
        for _ in 0..worker_count() {
            let _ = self.tx.send(Task::Shutdown);
        }
        let mut hs = self.workers.lock().unwrap();
        for h in hs.drain(..) {
            let _ = h.join();
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn pool_runs_jobs() {
        let sched = Scheduler::new();
        let n = Arc::new(AtomicUsize::new(0));
        for _ in 0..32 {
            let n = Arc::clone(&n);
            sched.submit(Box::new(move || {
                n.fetch_add(1, Ordering::Relaxed);
            }));
        }
        let start = std::time::Instant::now();
        while n.load(Ordering::Relaxed) < 32 && start.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(n.load(Ordering::Relaxed), 32);
        sched.shutdown();
    }
}
