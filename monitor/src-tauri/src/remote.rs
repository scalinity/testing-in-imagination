//! The only code in the monitor that reaches HiPerGator.
//!
//! Hard rules, enforced here and nowhere else:
//! 1. Every remote command is `cat`, `tail` or `ls` on a path that matches the allowlist built
//!    from monitor/DATA_CONTRACT.md. There is no way to build any other command: `RemoteRead`
//!    has private fields and one checked constructor per form.
//! 2. No Slurm command exists in this file, so none can be sent.
//! 3. The only transport is `ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false`
//!    through the existing master. `ProxyCommand=false` makes a fallback to a fresh connection
//!    fail instead of prompting for sign-in.
//! 4. A cycle starts only after `ssh -O check` (a local control command) succeeds; if it fails
//!    no ssh call is made and the last known state stays, with its age.
//! 5. Two cycles are never closer than 60 seconds, across restarts: the start time is written
//!    to disk before the first call.

use crate::config::{Config, MIN_REMOTE_INTERVAL_S};
use crate::util;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const HOST: &str = "daniel.escalante@hpg.rc.ufl.edu";
pub const ROOT: &str = "/blue/cai6734/daniel.escalante/private/tii-cee-1";
const SSH: &str = "/usr/bin/ssh";
const SSH_FIXED_ARGS: [&str; 6] = ["-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ProxyCommand=false"];

/// Directories that may be listed.
const LISTABLE: [&str; 4] = ["state", "state/progress", "gates", "export/approved"];
/// Single files that may be read.
const READABLE_FILES: [&str; 3] = ["state/status.json", "state/runlog.jsonl", "state/events.tsv"];
/// Directories whose `*.json` children may be read.
const READABLE_JSON_DIRS: [&str; 2] = ["state/progress", "gates"];

fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 120
        && !name.starts_with('.')
        && !name.starts_with('-')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn rel(path: &str) -> Option<&str> {
    path.strip_prefix(ROOT)?.strip_prefix('/')
}

fn listable(path: &str) -> bool {
    rel(path).is_some_and(|r| LISTABLE.contains(&r))
}

fn readable(path: &str) -> bool {
    let Some(r) = rel(path) else { return false };
    if READABLE_FILES.contains(&r) {
        return true;
    }
    READABLE_JSON_DIRS.iter().any(|dir| {
        r.strip_prefix(dir)
            .and_then(|rest| rest.strip_prefix('/'))
            .is_some_and(|name| safe_name(name) && name.ends_with(".json"))
    })
}

/// One permitted remote read. Fields are private: the constructors are the allowlist.
#[derive(Debug, Clone)]
pub struct RemoteRead {
    command: String,
    paths: Vec<String>,
    form: Form,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Form {
    Cat,
    TailFrom(u64),
    TailEach,
    Ls,
    LsRecursive,
}

impl RemoteRead {
    pub fn cat(path: &str) -> Result<RemoteRead, String> {
        Self::build(Form::Cat, vec![path.to_string()])
    }
    /// Bytes of an append-only file from a 0-based offset.
    pub fn tail_from(path: &str, offset: u64) -> Result<RemoteRead, String> {
        Self::build(Form::TailFrom(offset), vec![path.to_string()])
    }
    /// Whole contents of several small files, each under a `==> path <==` header.
    pub fn tail_each(paths: Vec<String>) -> Result<RemoteRead, String> {
        Self::build(Form::TailEach, paths)
    }
    pub fn ls(path: &str) -> Result<RemoteRead, String> {
        Self::build(Form::Ls, vec![path.to_string()])
    }
    pub fn ls_recursive(path: &str) -> Result<RemoteRead, String> {
        Self::build(Form::LsRecursive, vec![path.to_string()])
    }

    fn build(form: Form, paths: Vec<String>) -> Result<RemoteRead, String> {
        if paths.is_empty() || paths.len() > 64 {
            return Err("a remote read takes 1 to 64 paths".into());
        }
        for p in &paths {
            let allowed = match form {
                Form::Ls => listable(p),
                Form::LsRecursive => rel(p) == Some("export/approved"),
                Form::Cat | Form::TailFrom(_) | Form::TailEach => readable(p),
            };
            if !allowed {
                return Err(format!("path is not on the allowlist: {p}"));
            }
        }
        let joined = paths.join(" ");
        let command = match form {
            Form::Cat => format!("cat {joined}"),
            Form::TailFrom(offset) => format!("tail -c +{} {joined}", offset + 1),
            Form::TailEach => format!("tail -v -n +1 {joined}"),
            Form::Ls => format!("ls -ln --time-style=+%s {joined}"),
            Form::LsRecursive => format!("ls -lnR --time-style=+%s {joined}"),
        };
        // Belt and braces: the finished command holds nothing a shell could reinterpret.
        if !command
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '/' | '.' | '_' | '-' | '+' | '=' | '%'))
        {
            return Err("remote command holds an unexpected character".into());
        }
        Ok(RemoteRead { command, paths, form })
    }

    pub fn command(&self) -> &str {
        &self.command
    }

    fn ssh_args(&self) -> Vec<String> {
        let mut args: Vec<String> = SSH_FIXED_ARGS.iter().map(|s| s.to_string()).collect();
        args.push(HOST.to_string());
        args.push(self.command.clone());
        args
    }
}

/// `ssh -O check`: asks the local master socket whether it is alive. Opens no connection.
pub fn master_check() -> (bool, String) {
    let args = ["-O", "check", HOST].map(String::from);
    match util::run(SSH, &args, &[], None, Duration::from_secs(6), 4096) {
        Ok(out) if out.ok() => (true, first_line(&out.stderr, &out.text())),
        Ok(out) if out.timed_out => (false, "ssh -O check did not answer within 6 seconds".into()),
        Ok(out) => (false, first_line(&out.stderr, &out.text())),
        Err(e) => (false, format!("could not run ssh: {e}")),
    }
}

fn first_line(a: &str, b: &str) -> String {
    let s = if a.trim().is_empty() { b } else { a };
    util::one_line(s.lines().next().unwrap_or(""), 160)
}

pub enum Transport {
    Ssh,
    /// Demo mode: the same commands answered from a local directory that mirrors `ROOT`.
    Fixture(PathBuf),
}

pub struct ExecOut {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub ms: u64,
}

impl Transport {
    pub fn exec(&self, read: &RemoteRead) -> ExecOut {
        match self {
            Transport::Ssh => match util::run(SSH, &read.ssh_args(), &[], None, Duration::from_secs(25), 16 << 20) {
                Ok(out) => ExecOut {
                    code: if out.timed_out { None } else { out.code },
                    stderr: if out.timed_out { "timed out after 25 seconds".into() } else { out.stderr },
                    stdout: out.stdout,
                    ms: out.ms,
                },
                Err(e) => ExecOut { code: None, stdout: vec![], stderr: format!("could not run ssh: {e}"), ms: 0 },
            },
            Transport::Fixture(root) => fixture_exec(root, read),
        }
    }
}

fn fixture_exec(root: &Path, read: &RemoteRead) -> ExecOut {
    let local = |p: &str| root.join(rel(p).unwrap_or(""));
    let missing = |p: &str| ExecOut {
        code: Some(if matches!(read.form, Form::Ls | Form::LsRecursive) { 2 } else { 1 }),
        stdout: vec![],
        stderr: format!("cannot access '{p}': No such file or directory"),
        ms: 0,
    };
    let done = |stdout: Vec<u8>| ExecOut { code: Some(0), stdout, stderr: String::new(), ms: 0 };
    match read.form {
        Form::Cat => match std::fs::read(local(&read.paths[0])) {
            Ok(b) => done(b),
            Err(_) => missing(&read.paths[0]),
        },
        Form::TailFrom(offset) => match std::fs::read(local(&read.paths[0])) {
            Ok(b) => done(b.get(offset as usize..).unwrap_or(&[]).to_vec()),
            Err(_) => missing(&read.paths[0]),
        },
        Form::TailEach => {
            let mut out = Vec::new();
            for p in &read.paths {
                if let Ok(b) = std::fs::read(local(p)) {
                    if !out.is_empty() {
                        out.push(b'\n');
                    }
                    out.extend_from_slice(format!("==> {p} <==\n").as_bytes());
                    out.extend_from_slice(&b);
                }
            }
            done(out)
        }
        Form::Ls | Form::LsRecursive => {
            let dir = local(&read.paths[0]);
            if !dir.is_dir() {
                return missing(&read.paths[0]);
            }
            let mut out = String::new();
            fixture_ls(&dir, &read.paths[0], read.form == Form::LsRecursive, &mut out);
            done(out.into_bytes())
        }
    }
}

fn fixture_ls(dir: &Path, shown: &str, recursive: bool, out: &mut String) {
    if recursive {
        out.push_str(&format!("{shown}:\n"));
    }
    out.push_str("total 0\n");
    let mut subdirs = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten().collect();
    names.sort_by_key(|e| e.file_name());
    for entry in names {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(meta) = entry.metadata() else { continue };
        let epoch = util::mtime_ms(&meta).unwrap_or(0) / 1000;
        let kind = if meta.is_dir() { 'd' } else { '-' };
        out.push_str(&format!("{kind}rw------- 1 0 0 {} {epoch} {name}\n", meta.len()));
        if meta.is_dir() {
            subdirs.push(name);
        }
    }
    if recursive {
        for name in subdirs {
            out.push('\n');
            fixture_ls(&dir.join(&name), &format!("{shown}/{name}"), true, out);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LsEntry {
    pub dir: String,
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub mtime_ms: i64,
}

/// Parses `ls -ln --time-style=+%s` output, with or without `-R` headers.
pub fn parse_ls(text: &str, base: &str) -> Vec<LsEntry> {
    let mut dir = base.to_string();
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with("total ") {
            continue;
        }
        if let Some(header) = line.strip_suffix(':') {
            if header.starts_with('/') {
                dir = header.to_string();
                continue;
            }
        }
        let tok: Vec<&str> = line.split_whitespace().collect();
        if tok.len() < 7 {
            continue;
        }
        let (Ok(size), Ok(epoch)) = (tok[4].parse::<u64>(), tok[5].parse::<i64>()) else { continue };
        out.push(LsEntry {
            dir: dir.clone(),
            name: tok[6..].join(" "),
            is_dir: tok[0].starts_with('d'),
            size,
            mtime_ms: epoch * 1000,
        });
    }
    out
}

/// Splits `tail -v -n +1 a b` output into (path, contents).
pub fn parse_tail_each(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("==> ").and_then(|l| l.strip_suffix(" <==")) {
            out.push((path.to_string(), String::new()));
        } else if let Some(last) = out.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    out
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct CallLog {
    pub command: String,
    pub exit: Option<i32>,
    pub ms: u64,
    pub bytes: usize,
    pub note: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct EventRow {
    pub time: String,
    pub kind: String,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct JsonFile {
    pub name: String,
    pub size: u64,
    pub mtime_ms: i64,
    pub json: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ExportEntry {
    pub path: String,
    pub size: u64,
    pub mtime_ms: i64,
}

/// Everything known about the cluster. Cached to disk so that the last known state survives a
/// restart while the connection is down.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct RemoteData {
    /// never_read | ok | not_started | error
    pub phase: String,
    pub detail: String,
    pub last_attempt_ms: Option<i64>,
    pub last_ok_ms: Option<i64>,
    /// Set when a cycle was skipped because the SSH master was not running.
    pub skipped: Option<String>,
    pub skipped_ms: Option<i64>,
    pub status: Option<Value>,
    pub status_note: Option<String>,
    pub status_mtime_ms: Option<i64>,
    pub runlog: Vec<Value>,
    pub runlog_bad_lines: u32,
    pub events: Vec<EventRow>,
    pub progress: BTreeMap<String, JsonFile>,
    pub gates: BTreeMap<String, JsonFile>,
    pub exports: Vec<ExportEntry>,
    pub exports_listed: bool,
    pub calls: Vec<CallLog>,
    #[serde(default)]
    runlog_offset: u64,
    #[serde(default)]
    runlog_partial: String,
    #[serde(default)]
    events_offset: u64,
    #[serde(default)]
    events_partial: String,
}

const RUNLOG_KEEP: usize = 20_000;
const EVENTS_KEEP: usize = 2_000;

pub struct Remote {
    pub transport: Transport,
    pub data: RemoteData,
    pub last_start_ms: Option<i64>,
    cache_file: Option<PathBuf>,
    stamp_file: Option<PathBuf>,
}

impl Remote {
    pub fn new(cfg: &Config) -> Remote {
        if cfg.demo {
            return Remote {
                transport: Transport::Fixture(cfg.fixtures.join("hpg")),
                data: RemoteData { phase: "never_read".into(), ..Default::default() },
                last_start_ms: None,
                cache_file: None,
                stamp_file: None,
            };
        }
        let cache_file = cfg.data_dir.join("cluster-last-known.json");
        let stamp_file = cfg.data_dir.join("cluster-last-read-start");
        let data = std::fs::read_to_string(&cache_file)
            .ok()
            .and_then(|t| serde_json::from_str::<RemoteData>(&t).ok())
            .unwrap_or_else(|| RemoteData { phase: "never_read".into(), ..Default::default() });
        let last_start_ms = std::fs::read_to_string(&stamp_file).ok().and_then(|t| t.trim().parse().ok());
        Remote { transport: Transport::Ssh, data, last_start_ms, cache_file: Some(cache_file), stamp_file: Some(stamp_file) }
    }

    /// Seconds until a new cycle may start, given the configured interval.
    pub fn wait_s(&self, now_ms: i64, interval_s: u64) -> u64 {
        let floor = interval_s.max(MIN_REMOTE_INTERVAL_S) as i64 * 1000;
        match self.last_start_ms {
            Some(t) if now_ms - t < floor => ((floor - (now_ms - t)) as u64).div_ceil(1000),
            _ => 0,
        }
    }

    /// One read of the cluster. `master_up` is the result of `ssh -O check` taken just before.
    /// Returns false when nothing was attempted.
    pub fn cycle(&mut self, now_ms: i64, real_now_ms: i64, master: (bool, String)) -> bool {
        // The 60-second floor is measured on the real clock even in demo mode.
        if let Some(t) = self.last_start_ms {
            if real_now_ms - t < MIN_REMOTE_INTERVAL_S as i64 * 1000 {
                return false;
            }
        }
        if matches!(self.transport, Transport::Ssh) && !master.0 {
            self.data.skipped = Some(format!(
                "The SSH master is not running ({}), so no ssh call was made.",
                if master.1.is_empty() { "ssh -O check failed" } else { &master.1 }
            ));
            self.data.skipped_ms = Some(now_ms);
            return false;
        }
        self.last_start_ms = Some(real_now_ms);
        if let Some(stamp) = &self.stamp_file {
            if let Some(dir) = stamp.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(stamp, real_now_ms.to_string());
        }
        self.data.skipped = None;
        self.data.skipped_ms = None;
        self.data.last_attempt_ms = Some(now_ms);
        self.data.calls.clear();
        match self.read_all() {
            Ok(phase) => {
                self.data.phase = phase.0.into();
                self.data.detail = phase.1;
                self.data.last_ok_ms = Some(now_ms);
            }
            Err(e) => {
                self.data.phase = "error".into();
                self.data.detail = e;
            }
        }
        if let Some(cache) = &self.cache_file {
            if let Ok(text) = serde_json::to_string(&self.data) {
                let tmp = cache.with_extension("tmp");
                if std::fs::write(&tmp, text).is_ok() {
                    let _ = std::fs::rename(&tmp, cache);
                }
            }
        }
        true
    }

    fn call(&mut self, read: Result<RemoteRead, String>) -> Result<ExecOut, String> {
        let read = read?;
        let out = self.transport.exec(&read);
        self.data.calls.push(CallLog {
            command: read.command().to_string(),
            exit: out.code,
            ms: out.ms,
            bytes: out.stdout.len(),
            note: util::one_line(&out.stderr, 160),
        });
        // Exit 255 is ssh itself failing (the master vanished). Stop the whole cycle.
        if out.code.is_none() || out.code == Some(255) {
            return Err(format!("ssh could not complete `{}`: {}", read.command(), util::one_line(&out.stderr, 160)));
        }
        Ok(out)
    }

    fn read_all(&mut self) -> Result<(&'static str, String), String> {
        let p = |r: &str| format!("{ROOT}/{r}");

        let listing = self.call(RemoteRead::ls(&p("state")))?;
        if listing.code != Some(0) {
            return if listing.stderr.contains("No such file or directory") {
                Ok(("not_started", "state/ does not exist on HiPerGator yet. The watcher has not written anything.".into()))
            } else {
                Err(format!("could not list state/: {}", util::one_line(&listing.stderr, 160)))
            };
        }
        let state = parse_ls(&String::from_utf8_lossy(&listing.stdout), &p("state"));
        let find = |name: &str| state.iter().find(|e| e.name == name).cloned();

        match find("status.json") {
            Some(entry) => {
                let out = self.call(RemoteRead::cat(&p("state/status.json")))?;
                match serde_json::from_slice::<Value>(&out.stdout) {
                    Ok(v) if out.code == Some(0) => {
                        self.data.status = Some(v);
                        self.data.status_note = None;
                        self.data.status_mtime_ms = Some(entry.mtime_ms);
                    }
                    Ok(_) | Err(_) => {
                        self.data.status_note = Some("status.json could not be parsed on this read; showing the previous one.".into());
                    }
                }
            }
            None => {
                self.data.status = None;
                self.data.status_mtime_ms = None;
                self.data.status_note = Some("status.json has not been written yet.".into());
            }
        }

        if let Some(entry) = find("runlog.jsonl") {
            if entry.size < self.data.runlog_offset {
                self.data.runlog.clear();
                self.data.runlog_offset = 0;
                self.data.runlog_partial.clear();
                self.data.runlog_bad_lines = 0;
            }
            if entry.size > self.data.runlog_offset {
                let out = self.call(RemoteRead::tail_from(&p("state/runlog.jsonl"), self.data.runlog_offset))?;
                if out.code == Some(0) {
                    self.data.runlog_offset += out.stdout.len() as u64;
                    let mut text = std::mem::take(&mut self.data.runlog_partial);
                    text.push_str(&String::from_utf8_lossy(&out.stdout));
                    let complete = text.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    for line in text[..complete].lines().filter(|l| !l.trim().is_empty()) {
                        match serde_json::from_str::<Value>(line) {
                            Ok(v) if v.is_object() => self.data.runlog.push(v),
                            _ => self.data.runlog_bad_lines += 1,
                        }
                    }
                    self.data.runlog_partial = text[complete..].to_string();
                    let extra = self.data.runlog.len().saturating_sub(RUNLOG_KEEP);
                    self.data.runlog.drain(..extra);
                }
            }
        }

        if let Some(entry) = find("events.tsv") {
            if entry.size < self.data.events_offset {
                self.data.events.clear();
                self.data.events_offset = 0;
                self.data.events_partial.clear();
            }
            if entry.size > self.data.events_offset {
                let out = self.call(RemoteRead::tail_from(&p("state/events.tsv"), self.data.events_offset))?;
                if out.code == Some(0) {
                    self.data.events_offset += out.stdout.len() as u64;
                    let mut text = std::mem::take(&mut self.data.events_partial);
                    text.push_str(&String::from_utf8_lossy(&out.stdout));
                    let complete = text.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    for line in text[..complete].lines().filter(|l| !l.trim().is_empty()) {
                        let mut parts = line.splitn(3, '\t');
                        self.data.events.push(EventRow {
                            time: parts.next().unwrap_or("").to_string(),
                            kind: parts.next().unwrap_or("").to_string(),
                            text: util::one_line(parts.next().unwrap_or(""), 240),
                        });
                    }
                    self.data.events_partial = text[complete..].to_string();
                    let extra = self.data.events.len().saturating_sub(EVENTS_KEEP);
                    self.data.events.drain(..extra);
                }
            }
        }

        if find("progress").is_some_and(|e| e.is_dir) {
            let mut files = std::mem::take(&mut self.data.progress);
            let result = self.sync_json_dir("state/progress", &mut files);
            self.data.progress = files;
            result?;
        } else {
            self.data.progress.clear();
        }

        let mut gates = std::mem::take(&mut self.data.gates);
        let result = self.sync_json_dir("gates", &mut gates);
        self.data.gates = gates;
        result?;

        let exports = self.call(RemoteRead::ls_recursive(&p("export/approved")))?;
        if exports.code == Some(0) {
            let base = p("export/approved");
            self.data.exports = parse_ls(&String::from_utf8_lossy(&exports.stdout), &base)
                .into_iter()
                .filter(|e| !e.is_dir)
                .map(|e| ExportEntry {
                    path: format!("{}/{}", e.dir.strip_prefix(&base).unwrap_or("").trim_start_matches('/'), e.name)
                        .trim_start_matches('/')
                        .to_string(),
                    size: e.size,
                    mtime_ms: e.mtime_ms,
                })
                .collect();
            self.data.exports_listed = true;
        } else {
            self.data.exports.clear();
            self.data.exports_listed = false;
        }

        Ok(("ok", String::new()))
    }

    /// Lists a directory and reads the `*.json` files that are new or changed since last time.
    fn sync_json_dir(&mut self, dir: &str, files: &mut BTreeMap<String, JsonFile>) -> Result<(), String> {
        let base = format!("{ROOT}/{dir}");
        let listing = self.call(RemoteRead::ls(&base))?;
        if listing.code != Some(0) {
            files.clear();
            return Ok(());
        }
        let entries: Vec<LsEntry> = parse_ls(&String::from_utf8_lossy(&listing.stdout), &base)
            .into_iter()
            .filter(|e| !e.is_dir && e.name.ends_with(".json") && safe_name(&e.name) && e.size <= 1 << 20)
            .collect();
        files.retain(|name, _| entries.iter().any(|e| &e.name == name));
        let changed: Vec<&LsEntry> = entries
            .iter()
            .filter(|e| files.get(&e.name).is_none_or(|f| f.mtime_ms != e.mtime_ms || f.size != e.size))
            .collect();
        for chunk in changed.chunks(40) {
            let paths: Vec<String> = chunk.iter().map(|e| format!("{base}/{}", e.name)).collect();
            let out = self.call(RemoteRead::tail_each(paths))?;
            if out.code != Some(0) && out.stdout.is_empty() {
                continue;
            }
            for (path, body) in parse_tail_each(&String::from_utf8_lossy(&out.stdout)) {
                let Some(name) = path.strip_prefix(&base).map(|n| n.trim_start_matches('/')) else { continue };
                let Some(entry) = chunk.iter().find(|e| e.name == name) else { continue };
                files.insert(
                    name.to_string(),
                    JsonFile {
                        name: name.to_string(),
                        size: entry.size,
                        mtime_ms: entry.mtime_ms,
                        json: serde_json::from_str::<Value>(body.trim()).ok(),
                    },
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(r: &str) -> String {
        format!("{ROOT}/{r}")
    }

    #[test]
    fn contract_paths_are_allowed() {
        assert!(RemoteRead::cat(&p("state/status.json")).is_ok());
        assert!(RemoteRead::tail_from(&p("state/runlog.jsonl"), 0).is_ok());
        assert!(RemoteRead::tail_from(&p("state/events.tsv"), 120).is_ok());
        assert!(RemoteRead::tail_each(vec![p("state/progress/e4-cohort-x.json"), p("gates/G2a.json")]).is_ok());
        assert!(RemoteRead::ls(&p("state")).is_ok());
        assert!(RemoteRead::ls(&p("state/progress")).is_ok());
        assert!(RemoteRead::ls(&p("gates")).is_ok());
        assert!(RemoteRead::ls_recursive(&p("export/approved")).is_ok());
    }

    #[test]
    fn everything_else_is_refused() {
        let private = "/blue/cai6734/daniel.escalante/private/mimic-iv-ext-cdm/1.1/x.json";
        for bad in [
            private.to_string(),
            p("private/anything.json"),
            p("state/stages/e8.json"),
            p("state/../private/x.json"),
            p("state/progress/../../private/x.json"),
            p("state/progress/.hidden.json"),
            p("state/progress/a b.json"),
            p("state/progress/a;squeue.json"),
            p("state/progress/$(squeue).json"),
            p("state/progress/x.txt"),
            p("state/watcher_1.log"),
            p("pipeline/graph.json"),
            p("export/approved/E1/result.json"),
            "/etc/passwd".to_string(),
            "state/status.json".to_string(),
            format!("{ROOT}-other/state/status.json"),
        ] {
            assert!(RemoteRead::cat(&bad).is_err(), "cat accepted {bad}");
            assert!(RemoteRead::tail_from(&bad, 0).is_err(), "tail accepted {bad}");
            assert!(RemoteRead::tail_each(vec![bad.clone()]).is_err(), "tail accepted {bad}");
        }
        for bad in [p("private"), p(""), ROOT.to_string(), p("state/stages"), p("export"), "/blue/cai6734".to_string()] {
            assert!(RemoteRead::ls(&bad).is_err(), "ls accepted {bad}");
            assert!(RemoteRead::ls_recursive(&bad).is_err(), "ls -R accepted {bad}");
        }
        assert!(RemoteRead::ls_recursive(&p("state")).is_err());
    }

    #[test]
    fn only_cat_tail_and_ls_are_ever_sent_and_never_slurm() {
        let reads = [
            RemoteRead::cat(&p("state/status.json")).unwrap(),
            RemoteRead::tail_from(&p("state/runlog.jsonl"), 41).unwrap(),
            RemoteRead::tail_each(vec![p("gates/G0.json")]).unwrap(),
            RemoteRead::ls(&p("gates")).unwrap(),
            RemoteRead::ls_recursive(&p("export/approved")).unwrap(),
        ];
        for r in &reads {
            let verb = r.command().split(' ').next().unwrap();
            assert!(["cat", "tail", "ls"].contains(&verb), "unexpected verb in {}", r.command());
            for slurm in ["squeue", "sacct", "sbatch", "scontrol", "sinfo", "scancel", "srun", "sacctmgr", "scrontab"] {
                assert!(!r.command().contains(slurm));
            }
            let args = r.ssh_args();
            assert_eq!(&args[..6], &["-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ProxyCommand=false"]);
            assert_eq!(args[6], HOST);
            assert_eq!(args.len(), 8);
        }
        assert_eq!(reads[1].command(), format!("tail -c +42 {}", p("state/runlog.jsonl")));
    }

    #[test]
    fn parses_ls_and_tail_output() {
        let text = "total 12\n-rw------- 1 1234 567 2048 1794557702 status.json\ndrwx--S--- 2 1234 567 4096 1794557000 progress\n";
        let entries = parse_ls(text, "/x/state");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "status.json");
        assert_eq!(entries[0].size, 2048);
        assert_eq!(entries[0].mtime_ms, 1_794_557_702_000);
        assert!(entries[1].is_dir);

        let recursive = "/x/export/approved:\ntotal 4\ndrwx------ 2 1 1 4096 1794557000 E1\n\n/x/export/approved/E1:\ntotal 4\n-rw------- 1 1 1 10 1794557001 summary.json\n";
        let entries = parse_ls(recursive, "/x/export/approved");
        assert_eq!(entries[1].dir, "/x/export/approved/E1");
        assert_eq!(entries[1].name, "summary.json");

        let tail = "==> /a/one.json <==\n{\"a\": 1}\n\n==> /a/two.json <==\n{\"b\": 2}\n";
        let parts = parse_tail_each(tail);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].0, "/a/two.json");
        assert_eq!(parts[1].1.trim(), "{\"b\": 2}");
    }

    #[test]
    fn the_sixty_second_floor_holds_whatever_the_setting() {
        let mut remote = Remote {
            transport: Transport::Fixture(PathBuf::from("/nonexistent")),
            data: RemoteData::default(),
            last_start_ms: Some(1_000_000),
            cache_file: None,
            stamp_file: None,
        };
        assert_eq!(remote.wait_s(1_000_000 + 10_000, 5), 50);
        assert_eq!(remote.wait_s(1_000_000 + 10_000, 120), 110);
        assert_eq!(remote.wait_s(1_000_000 + 130_000, 120), 0);
        assert!(!remote.cycle(1_030_000, 1_030_000, (true, String::new())));
        assert!(remote.data.calls.is_empty());
    }

    /// A running stage's progress file and the append-only run log are followed from one read
    /// to the next: changed files are fetched again, unchanged ones are not, and only the
    /// appended bytes of the run log are asked for.
    #[test]
    fn progress_and_run_log_follow_the_files_between_reads() {
        let root = std::env::temp_dir().join(format!("tii-monitor-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for dir in ["state/progress", "gates", "export/approved"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        let write = |rel: &str, text: &str, epoch_s: u64| {
            let path = root.join(rel);
            std::fs::write(&path, text).unwrap();
            let file = std::fs::File::options().write(true).open(&path).unwrap();
            file.set_modified(std::time::UNIX_EPOCH + Duration::from_secs(epoch_s)).unwrap();
        };
        write("state/status.json", r#"{"schema":"tii-status/1"}"#, 1000);
        write("state/runlog.jsonl", "{\"event\":\"submitted\",\"stage\":\"s1\"}\n", 1000);
        write("state/progress/s1.json", r#"{"stage":"s1","done":10,"total":100}"#, 1000);

        let up = || (true, String::new());
        let mut remote = Remote { transport: Transport::Fixture(root.clone()), data: RemoteData::default(), last_start_ms: None, cache_file: None, stamp_file: None };
        assert!(remote.cycle(1, 1_000_000, up()));
        assert_eq!(remote.data.phase, "ok");
        assert_eq!(remote.data.runlog.len(), 1);
        assert_eq!(remote.data.progress["s1.json"].json.as_ref().unwrap()["done"], 10);
        let first_read = remote.data.calls.len();

        // Nothing changed: the next read lists and re-reads status.json, and fetches nothing else.
        assert!(remote.cycle(2, 1_061_000, up()));
        assert!(remote.data.calls.len() < first_read);
        assert!(!remote.data.calls.iter().any(|c| c.command.starts_with("tail")));
        assert_eq!(remote.data.runlog.len(), 1);

        // The stage reports more progress and the watcher appends a line.
        write("state/progress/s1.json", r#"{"stage":"s1","done":55,"total":100}"#, 1060);
        let mut log = std::fs::read_to_string(root.join("state/runlog.jsonl")).unwrap();
        log.push_str("{\"event\":\"started\",\"stage\":\"s1\"}\n");
        write("state/runlog.jsonl", &log, 1060);
        assert!(remote.cycle(3, 1_122_000, up()));
        assert_eq!(remote.data.progress["s1.json"].json.as_ref().unwrap()["done"], 55);
        assert_eq!(remote.data.runlog.len(), 2);
        assert_eq!(remote.data.runlog[1]["event"], "started");
        assert!(remote.data.calls.iter().any(|c| c.command.starts_with("tail -c +") && !c.command.starts_with("tail -c +1 ")));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_dead_master_means_no_call() {
        let mut remote = Remote {
            transport: Transport::Ssh,
            data: RemoteData::default(),
            last_start_ms: None,
            cache_file: None,
            stamp_file: None,
        };
        assert!(!remote.cycle(5_000_000, 5_000_000, (false, "No such file or directory".into())));
        assert!(remote.data.calls.is_empty());
        assert!(remote.data.skipped.is_some());
        assert!(remote.last_start_ms.is_none());
    }
}
