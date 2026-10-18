//! Refcounted struct/map cells with weak back-edges for cycle breaking.

use crate::interpreter::Value;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

pub type FieldMap = HashMap<String, Value>;
pub type ObjectCell = Arc<Mutex<FieldMap>>;
pub type WeakCell = Weak<Mutex<FieldMap>>;

thread_local! {
    static DISPLAYING: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
}

fn cell_ptr(cell: &ObjectCell) -> usize {
    Arc::as_ptr(cell) as usize
}

fn reaches(val: &Value, target: usize, visiting: &mut HashSet<usize>) -> bool {
    match val {
        Value::Struct(_, cell) | Value::Map(cell) => {
            let p = cell_ptr(cell);
            if p == target {
                return true;
            }
            if !visiting.insert(p) {
                return false;
            }
            let guard = cell.lock().unwrap();
            for v in guard.values() {
                if reaches(v, target, visiting) {
                    return true;
                }
            }
            false
        }
        Value::List(items) | Value::Enum(_, _, items) => items.iter().any(|v| reaches(v, target, visiting)),
        Value::Message { args, .. } => args.iter().any(|v| reaches(v, target, visiting)),
        Value::Future(inner) => reaches(inner, target, visiting),
        _ => false,
    }
}

fn weaken(val: Value) -> Value {
    match val {
        Value::Struct(name, cell) => Value::WeakStruct(name, Arc::downgrade(&cell)),
        Value::Map(cell) => Value::WeakMap(Arc::downgrade(&cell)),
        Value::List(items) => Value::List(items.into_iter().map(weaken).collect()),
        other => other,
    }
}

/// Store `val` on `host`, using a weak back-edge if that would form a cycle.
pub fn install_field(host: &ObjectCell, key: String, val: Value) {
    let target = cell_ptr(host);
    let stored = if reaches(&val, target, &mut HashSet::new()) {
        weaken(val)
    } else {
        val
    };
    host.lock().unwrap().insert(key, stored);
}

pub fn new_struct(name: String, fields: FieldMap) -> Value {
    let cell: ObjectCell = Arc::new(Mutex::new(HashMap::new()));
    for (k, v) in fields {
        install_field(&cell, k, v);
    }
    Value::Struct(name, cell)
}

pub fn new_map(fields: FieldMap) -> Value {
    let cell: ObjectCell = Arc::new(Mutex::new(HashMap::new()));
    for (k, v) in fields {
        install_field(&cell, k, v);
    }
    Value::Map(cell)
}

/// Drop nested strong maps so remaining Arcs can hit zero after a scope ends.
pub fn dissolve(val: &Value) {
    match val {
        Value::Struct(_, cell) | Value::Map(cell) => {
            let drained = {
                let mut g = cell.lock().unwrap();
                std::mem::take(&mut *g)
            };
            for v in drained.values() {
                dissolve(v);
            }
        }
        Value::List(items) | Value::Enum(_, _, items) => {
            for v in items {
                dissolve(v);
            }
        }
        Value::Future(inner) => dissolve(inner),
        _ => {}
    }
}

pub fn display_value(val: &Value, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    DISPLAYING.with(|vis| fmt_value(val, f, &mut vis.borrow_mut()))
}

fn fmt_value(val: &Value, f: &mut fmt::Formatter<'_>, visiting: &mut HashSet<usize>) -> fmt::Result {
    match val {
        Value::Int(n) => write!(f, "{}", n),
        Value::Float(x) => write!(f, "{}", x),
        Value::Str(s) => write!(f, "{}", s),
        Value::Bool(b) => write!(f, "{}", b),
        Value::List(items) => {
            write!(f, "[")?;
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_value(v, f, visiting)?;
            }
            write!(f, "]")
        }
        Value::Tensor { data, shape } => write!(f, "tensor(shape={:?}, data={:?})", shape, data),
        Value::Message { tag, args } => {
            write!(f, "{}(", tag)?;
            for (i, v) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_value(v, f, visiting)?;
            }
            write!(f, ")")
        }
        Value::ActorHandle(id) => write!(f, "<actor #{}>", id),
        Value::Struct(name, cell) => fmt_cell(f, visiting, Some(name), cell),
        Value::Map(cell) => fmt_cell(f, visiting, None, cell),
        Value::WeakStruct(name, _) => write!(f, "{} {{ <weak> }}", name),
        Value::WeakMap(_) => write!(f, "{{ <weak> }}"),
        Value::Enum(type_name, variant, args) => {
            if args.is_empty() {
                write!(f, "{}.{}", type_name, variant)
            } else {
                write!(f, "{}.{}(", type_name, variant)?;
                for (i, v) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_value(v, f, visiting)?;
                }
                write!(f, ")")
            }
        }
        Value::Closure(_) => write!(f, "<closure>"),
        Value::Future(inner) => {
            write!(f, "<future: ")?;
            fmt_value(inner, f, visiting)?;
            write!(f, ">")
        }
        Value::Unit => write!(f, "()"),
    }
}

fn fmt_cell(
    f: &mut fmt::Formatter<'_>,
    visiting: &mut HashSet<usize>,
    name: Option<&str>,
    cell: &ObjectCell,
) -> fmt::Result {
    let p = cell_ptr(cell);
    if !visiting.insert(p) {
        return match name {
            Some(n) => write!(f, "{} {{ .. }}", n),
            None => write!(f, "{{ .. }}"),
        };
    }
    let guard = cell.lock().unwrap();
    let mut parts: Vec<(String, String)> = Vec::new();
    for (k, v) in guard.iter() {
        let mut buf = String::new();
        {
            use std::fmt::Write as _;
            let _ = write!(&mut buf, "{}", Fmt(v, RefCell::new(visiting)));
        }
        parts.push((k.clone(), buf));
    }
    drop(guard);
    parts.sort_by(|a, b| a.0.cmp(&b.0));
    visiting.remove(&p);
    match name {
        Some(n) => {
            write!(f, "{} {{ ", n)?;
            for (i, (k, v)) in parts.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", k, v)?;
            }
            write!(f, " }}")
        }
        None => {
            write!(f, "{{ ")?;
            for (i, (k, v)) in parts.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "\"{}\": {}", k, v)?;
            }
            write!(f, " }}")
        }
    }
}

struct Fmt<'a>(&'a Value, RefCell<&'a mut HashSet<usize>>);

impl fmt::Display for Fmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_value(self.0, f, &mut self.1.borrow_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_does_not_pin_refcount() {
        let a = new_struct("Node".into(), HashMap::new());
        let Value::Struct(_, cell) = &a else { panic!() };
        install_field(cell, "self".into(), a.clone());
        assert!(matches!(
            cell.lock().unwrap().get("self"),
            Some(Value::WeakStruct(_, _))
        ));
        let weak = match cell.lock().unwrap().get("self") {
            Some(Value::WeakStruct(_, w)) => w.clone(),
            _ => panic!(),
        };
        drop(a);
        assert!(weak.upgrade().is_none());
    }
}
