use std::time::Instant;
use tridentix_actor_rt::{MailboxTable, channel};

fn main() {
    let table = MailboxTable::<i64>::new();
    let (tx, rx) = channel();
    table.register(0, tx);

    const N: u32 = 50_000;
    let start = Instant::now();
    for i in 0..N {
        table.send(0, i as i64);
        let _ = rx.recv();
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() / N as u128;
    println!("ping-pong dispatch: {ns} ns/msg (N={N})");
    println!("zero-gc path: bounded MPSC, no tracing collector");
}
