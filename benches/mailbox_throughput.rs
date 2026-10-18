use std::time::Instant;
use tridentix_actor_rt::{MailboxTable, channel, MAILBOX_BOUND};

fn main() {
    let table = MailboxTable::<i64>::new();
    let (tx, rx) = channel();
    table.register(1, tx);

    const N: u64 = 200_000;
    let start = Instant::now();
    let mut sent = 0u64;
    while sent < N {
        table.send(1, sent as i64);
        sent += 1;
        while rx.try_recv().is_ok() {}
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    println!("throughput: {:.1} msg/s (bound={MAILBOX_BOUND})", sent as f64 / secs);
}
