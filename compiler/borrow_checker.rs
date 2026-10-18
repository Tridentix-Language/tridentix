//! Tensor ownership and exclusive borrows (many shared XOR one mutable).

use crate::ast::Expr;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorState {
    Owned,
    Moved,
}

pub struct BorrowChecker {
    tensor_scopes: Vec<HashMap<String, TensorState>>,
    borrow_scopes: Vec<Vec<(String, bool)>>,
    active_borrows: HashMap<String, (u32, bool)>,
}

impl BorrowChecker {
    pub fn new() -> Self {
        BorrowChecker {
            tensor_scopes: vec![HashMap::new()],
            borrow_scopes: vec![Vec::new()],
            active_borrows: HashMap::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.tensor_scopes.push(HashMap::new());
        self.borrow_scopes.push(Vec::new());
    }

    pub fn pop_scope(&mut self) {
        self.tensor_scopes.pop();
        if let Some(released) = self.borrow_scopes.pop() {
            for (source, is_mut) in released {
                self.release_borrow(&source, is_mut);
            }
        }
    }

    pub fn declare_owned(&mut self, name: &str) {
        self.tensor_scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), TensorState::Owned);
    }

    pub fn is_tensor(&self, name: &str) -> bool {
        self.tensor_scopes.iter().rev().any(|s| s.contains_key(name))
    }

    pub fn is_tensor_ident(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Ident(name) if self.is_tensor(name))
    }

    fn tensor_state(&self, name: &str) -> Option<TensorState> {
        for scope in self.tensor_scopes.iter().rev() {
            if let Some(state) = scope.get(name) {
                return Some(state.clone());
            }
        }
        None
    }

    fn mark_moved(&mut self, name: &str) {
        for scope in self.tensor_scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), TensorState::Moved);
                return;
            }
        }
    }

    pub fn has_active_borrow(&self, name: &str) -> bool {
        matches!(self.active_borrows.get(name), Some((count, is_mut)) if *count > 0 || *is_mut)
    }

    pub fn check_tensor_move(&mut self, expr: &Expr, context: &str) -> Option<String> {
        let Expr::Ident(name) = expr else {
            return None;
        };
        match self.tensor_state(name) {
            Some(TensorState::Moved) => Some(format!(
                "use of moved tensor '{}' as {} — it was already moved to a new owner earlier",
                name, context
            )),
            Some(TensorState::Owned) => {
                if self.has_active_borrow(name) {
                    Some(format!(
                        "cannot move '{}' as {} while it is still borrowed",
                        name, context
                    ))
                } else {
                    self.mark_moved(name);
                    None
                }
            }
            None => None,
        }
    }

    pub fn check_borrow_conflict(&mut self, source: &str, want_mut: bool) -> Option<String> {
        let (immut_count, mut_active) = self.active_borrows.get(source).copied().unwrap_or((0, false));
        if want_mut {
            if immut_count > 0 || mut_active {
                return Some(format!(
                    "cannot borrow_mut('{}') — it is already borrowed elsewhere",
                    source
                ));
            }
        } else if mut_active {
            return Some(format!(
                "cannot borrow('{}') — it is currently mutably borrowed elsewhere",
                source
            ));
        }
        None
    }

    pub fn register_persistent_borrow(&mut self, source: &str, is_mut: bool) {
        let entry = self.active_borrows.entry(source.to_string()).or_insert((0, false));
        if is_mut {
            entry.1 = true;
        } else {
            entry.0 += 1;
        }
        self.borrow_scopes
            .last_mut()
            .unwrap()
            .push((source.to_string(), is_mut));
    }

    fn release_borrow(&mut self, source: &str, is_mut: bool) {
        if let Some(entry) = self.active_borrows.get_mut(source) {
            if is_mut {
                entry.1 = false;
            } else {
                entry.0 = entry.0.saturating_sub(1);
            }
        }
    }
}

impl Default for BorrowChecker {
    fn default() -> Self {
        Self::new()
    }
}
