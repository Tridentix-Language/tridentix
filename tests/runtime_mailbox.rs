use tridentix_actor_rt::mailbox::{channel, MailboxTable, MAILBOX_BOUND};
use tridentix_actor_rt::supervisor::{restart_set, RestartStrategy};

#[test]
fn mailbox_bound_is_fixed() {
    let table = MailboxTable::<i64>::new();
    let (tx, rx) = channel();
    table.register(0, tx);
    for i in 0..(MAILBOX_BOUND as i64 + 8) {
        table.send(0, i);
    }
    let mut n = 0;
    while rx.try_recv().is_ok() {
        n += 1;
    }
    assert_eq!(n, MAILBOX_BOUND);
}

#[test]
fn one_for_all_restarts_every_child() {
    assert_eq!(restart_set(RestartStrategy::OneForAll, 2, 3), vec![0, 1, 2]);
}
