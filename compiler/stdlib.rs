//! Native stdlib: files, JSON, regex, TCP/UDP, HTTP.

use crate::interpreter::{RuntimeError, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

type RResult<T> = Result<T, RuntimeError>;

fn type_err(msg: impl Into<String>) -> RuntimeError {
    RuntimeError::TypeError(msg.into())
}

// File I/O

pub fn file_read(path: &str) -> RResult<Value> {
    fs::read_to_string(path)
        .map(Value::Str)
        .map_err(|e| type_err(format!("file_read('{}'): {}", path, e)))
}

pub fn file_write(path: &str, content: &str) -> RResult<Value> {
    fs::write(path, content)
        .map(|_| Value::Bool(true))
        .map_err(|e| type_err(format!("file_write('{}'): {}", path, e)))
}

pub fn file_append(path: &str, content: &str) -> RResult<Value> {
    use std::fs::OpenOptions;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| type_err(format!("file_append('{}'): {}", path, e)))?;
    f.write_all(content.as_bytes())
        .map_err(|e| type_err(format!("file_append('{}'): {}", path, e)))?;
    Ok(Value::Bool(true))
}

pub fn file_exists(path: &str) -> Value {
    Value::Bool(std::path::Path::new(path).exists())
}

// JSON via serde_json, mapped to Value::Struct / Value::List.

pub fn json_stringify(v: &Value) -> RResult<Value> {
    let json_val = value_to_json(v)?;
    serde_json::to_string(&json_val)
        .map(Value::Str)
        .map_err(|e| type_err(format!("json_stringify(): {}", e)))
}

pub fn json_parse(s: &str) -> RResult<Value> {
    let parsed: serde_json::Value =
        serde_json::from_str(s).map_err(|e| type_err(format!("json_parse(): invalid JSON: {}", e)))?;
    Ok(json_to_value(&parsed))
}

fn value_to_json(v: &Value) -> RResult<serde_json::Value> {
    use serde_json::Value as J;
    Ok(match v {
        Value::Int(n) => J::Number((*n).into()),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(J::Number).unwrap_or(J::Null),
        Value::Str(s) => J::String(s.clone()),
        Value::Bool(b) => J::Bool(*b),
        Value::List(items) => {
            let converted: RResult<Vec<J>> = items.iter().map(value_to_json).collect();
            J::Array(converted?)
        }
        Value::Struct(_, fields) => {
            let guard = fields.lock().unwrap();
            let mut map = serde_json::Map::new();
            for (k, fv) in guard.iter() {
                map.insert(k.clone(), value_to_json(fv)?);
            }
            J::Object(map)
        }
        Value::Unit => J::Null,
        other => return Err(type_err(format!("json_stringify(): unsupported value {}", other))),
    })
}

fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(items) => Value::List(items.iter().map(json_to_value).collect()),
        serde_json::Value::Object(map) => {
            let mut hm = std::collections::HashMap::new();
            for (k, v) in map.iter() {
                hm.insert(k.clone(), json_to_value(v));
            }
            Value::Struct("json_object".to_string(), std::sync::Arc::new(std::sync::Mutex::new(hm)))
        }
    }
}

// Regex via the `regex` crate.

pub fn regex_match(pattern: &str, text: &str) -> RResult<Value> {
    let re = regex::Regex::new(pattern).map_err(|e| type_err(format!("regex_match(): invalid pattern: {}", e)))?;
    Ok(Value::Bool(re.is_match(text)))
}

pub fn regex_find(pattern: &str, text: &str) -> RResult<Value> {
    let re = regex::Regex::new(pattern).map_err(|e| type_err(format!("regex_find(): invalid pattern: {}", e)))?;
    Ok(match re.find(text) {
        Some(m) => Value::Str(m.as_str().to_string()),
        None => Value::Str(String::new()),
    })
}

pub fn regex_replace(pattern: &str, text: &str, replacement: &str) -> RResult<Value> {
    let re = regex::Regex::new(pattern).map_err(|e| type_err(format!("regex_replace(): invalid pattern: {}", e)))?;
    Ok(Value::Str(re.replace_all(text, replacement).to_string()))
}

// TCP client via std::net.

pub fn net_tcp_send(host: &str, port: i64, message: &str) -> RResult<Value> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr).map_err(|e| type_err(format!("net_tcp_send('{}'): connect failed: {}", addr, e)))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| type_err(format!("net_tcp_send(): {}", e)))?;
    stream
        .write_all(message.as_bytes())
        .map_err(|e| type_err(format!("net_tcp_send('{}'): write failed: {}", addr, e)))?;

    let mut buf = [0u8; 4096];
    let n = stream
        .read(&mut buf)
        .map_err(|e| type_err(format!("net_tcp_send('{}'): read failed: {}", addr, e)))?;
    Ok(Value::Str(String::from_utf8_lossy(&buf[..n]).to_string()))
}

// Subprocess execution via std::process.

pub fn process_run(cmd: &str, args: &[String]) -> RResult<Value> {
    let output = std::process::Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| type_err(format!("process_run('{}'): {}", cmd, e)))?;

    let mut result = std::collections::HashMap::new();
    result.insert("stdout".to_string(), Value::Str(String::from_utf8_lossy(&output.stdout).to_string()));
    result.insert("stderr".to_string(), Value::Str(String::from_utf8_lossy(&output.stderr).to_string()));
    result.insert(
        "exit_code".to_string(),
        Value::Int(output.status.code().unwrap_or(-1) as i64),
    );
    Ok(Value::Struct(
        "ProcessResult".to_string(),
        std::sync::Arc::new(std::sync::Mutex::new(result)),
    ))
}

// UDP

pub fn udp_send(host: &str, port: i64, message: &str) -> RResult<Value> {
    use std::net::UdpSocket;
    let addr = format!("{}:{}", host, port);
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| type_err(format!("udp_send(): couldn't bind local socket: {}", e)))?;
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| type_err(format!("udp_send(): {}", e)))?;
    socket
        .send_to(message.as_bytes(), &addr)
        .map_err(|e| type_err(format!("udp_send('{}'): send failed: {}", addr, e)))?;

    let mut buf = [0u8; 4096];
    match socket.recv_from(&mut buf) {
        Ok((n, _)) => Ok(Value::Str(String::from_utf8_lossy(&buf[..n]).to_string())),
        // Timeout with no reply returns empty string (UDP is unreliable).
        Err(_) => Ok(Value::Str(String::new())),
    }
}

// HTTP/1.1 GET/POST over TcpStream (plain HTTP).

fn http_request(method: &str, host: &str, port: i64, path: &str, body: Option<&str>) -> RResult<Value> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr).map_err(|e| type_err(format!("http request to '{}': connect failed: {}", addr, e)))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| type_err(format!("http request: {}", e)))?;

    let body_bytes = body.unwrap_or("");
    let request = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        method,
        path,
        host,
        body_bytes.len(),
        body_bytes
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| type_err(format!("http request: write failed: {}", e)))?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| type_err(format!("http request: read failed: {}", e)))?;
    let response_str = String::from_utf8_lossy(&response);

    // Status line + body after the header/body blank line.
    let (status_line, rest) = response_str.split_once("\r\n").unwrap_or((&response_str, ""));
    let status_code: i64 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let resp_body = rest.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or(rest);

    let mut result = std::collections::HashMap::new();
    result.insert("status".to_string(), Value::Int(status_code));
    result.insert("body".to_string(), Value::Str(resp_body.to_string()));
    Ok(Value::Struct("HttpResponse".to_string(), std::sync::Arc::new(std::sync::Mutex::new(result))))
}

pub fn http_get(host: &str, port: i64, path: &str) -> RResult<Value> {
    http_request("GET", host, port, path, None)
}

pub fn http_post(host: &str, port: i64, path: &str, body: &str) -> RResult<Value> {
    http_request("POST", host, port, path, Some(body))
}

