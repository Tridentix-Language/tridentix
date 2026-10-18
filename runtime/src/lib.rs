//! Native actor runtime: scheduler, bounded mailbox, supervisor, C ABI for LLVM 20 AOT/JIT.

pub mod mailbox;
pub mod scheduler;
pub mod supervisor;

pub use mailbox::{channel, MailboxTable, ACTOR_STACK, MAILBOX_BOUND};
pub use scheduler::Scheduler;
pub use supervisor::{restart_set, RestartBudget, RestartStrategy};

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

fn mailboxes() -> &'static MailboxTable<i64> {
    static MAILBOXES: OnceLock<MailboxTable<i64>> = OnceLock::new();
    MAILBOXES.get_or_init(MailboxTable::new)
}

fn next_id() -> &'static AtomicI64 {
    static NEXT_ID: OnceLock<AtomicI64> = OnceLock::new();
    NEXT_ID.get_or_init(|| AtomicI64::new(0))
}

fn handles() -> &'static Mutex<Vec<JoinHandle<()>>> {
    static HANDLES: OnceLock<Mutex<Vec<JoinHandle<()>>>> = OnceLock::new();
    HANDLES.get_or_init(|| Mutex::new(Vec::new()))
}

const SHUTDOWN_SENTINEL: i64 = i64::MIN;

pub type ActorHandlerFn = extern "C" fn(i64) -> i64;

#[no_mangle]
pub extern "C" fn tridentix_rt_spawn(handler: ActorHandlerFn) -> i64 {
    let id = next_id().fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = mailbox::channel();
    mailboxes().register(id as usize, tx);

    let join_handle = match Scheduler::global().spawn_blocking(
        &format!("tridentix-actor-{id}"),
        mailbox::ACTOR_STACK,
        move || {
            while let Ok(msg) = rx.recv() {
                if msg == SHUTDOWN_SENTINEL {
                    break;
                }
                handler(msg);
            }
        },
    ) {
        Ok(h) => h,
        Err(_) => return -1,
    };

    handles().lock().unwrap().push(join_handle);
    id
}

#[no_mangle]
pub extern "C" fn tridentix_rt_send(actor_id: i64, payload: i64) {
    if actor_id < 0 {
        return;
    }
    mailboxes().send(actor_id as usize, payload);
}

#[no_mangle]
pub extern "C" fn tridentix_rt_shutdown_and_join() {
    mailboxes().for_each_sender(|sender| {
        let _ = sender.send(SHUTDOWN_SENTINEL);
    });
    let mut hs = handles().lock().unwrap();
    for h in hs.drain(..) {
        let _ = h.join();
    }
}
