use tridentix_actor_rt::mailbox::{channel, MailboxTable, SendStatus, MAILBOX_BOUND};
use tridentix_actor_rt::supervisor::{restart_set, RestartStrategy};

#[test]
fn mailbox_bound_is_fixed() {
    let table = MailboxTable::<i64>::new();
    let (tx, rx) = channel();
    table.register(0, tx);
    for i in 0..(MAILBOX_BOUND as i64 + 8) {
        let status = table.send(0, i);
        if (i as usize) < MAILBOX_BOUND {
            assert_eq!(status, SendStatus::Delivered);
        } else {
            assert_eq!(status, SendStatus::Full);
        }
    }
    let mut n = 0;
    while rx.try_recv().is_ok() {
        n += 1;
    }
    assert_eq!(n, MAILBOX_BOUND);
}

#[test]
fn send_status_reports_unregistered_actor() {
    let table = MailboxTable::<i64>::new();
    assert_eq!(table.send(42, 100), SendStatus::NoSuchActor);
}

#[test]
fn send_status_reports_disconnected_actor() {
    let table = MailboxTable::<i64>::new();
    let (tx, rx) = channel();
    table.register(0, tx);
    drop(rx);
    assert_eq!(table.send(0, 100), SendStatus::Disconnected);
}

#[test]
fn one_for_all_restarts_every_child() {
    assert_eq!(restart_set(RestartStrategy::OneForAll, 2, 3), vec![0, 1, 2]);
}
