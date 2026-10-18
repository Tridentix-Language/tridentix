//! Native supervisor policies (one-for-one / one-for-all / rest-for-one).

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    OneForOne,
    OneForAll,
    RestForOne,
}

#[derive(Debug, Clone)]
pub struct RestartBudget {
    pub max_restarts: i64,
    pub window: Duration,
    count: i64,
    window_start: Instant,
}

impl RestartBudget {
    pub fn new(max_restarts: i64, window: Duration) -> Self {
        RestartBudget {
            max_restarts,
            window,
            count: 0,
            window_start: Instant::now(),
        }
    }

    pub fn allow(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.window_start) > self.window {
            self.window_start = now;
            self.count = 0;
        }
        if self.count >= self.max_restarts {
            return false;
        }
        self.count += 1;
        true
    }
}

pub fn restart_set(strategy: RestartStrategy, crashed: usize, child_count: usize) -> Vec<usize> {
    match strategy {
        RestartStrategy::OneForOne => vec![crashed],
        RestartStrategy::OneForAll => (0..child_count).collect(),
        RestartStrategy::RestForOne => (crashed..child_count).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_for_one_restarts_tail() {
        assert_eq!(restart_set(RestartStrategy::RestForOne, 1, 4), vec![1, 2, 3]);
    }

    #[test]
    fn budget_trips() {
        let mut b = RestartBudget::new(2, Duration::from_secs(60));
        assert!(b.allow());
        assert!(b.allow());
        assert!(!b.allow());
    }
}
