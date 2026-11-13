//! Claude Code transcripts: what each AI worker is doing.
//!
//! Each transcript is a JSONL file that only grows. A line may be several megabytes (request
//! blobs, tool outputs), so lines are deserialized into structs that name only the fields needed;
//! serde skips everything else without allocating it. Tool outputs are never kept: from a
//! `tool_result` only its id, its error flag and the line's time are read.

use crate::config::Config;
use crate::util;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Line {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    message: Option<Msg>,
    #[serde(rename = "aiTitle", default)]
    ai_title: Option<String>,
}

#[derive(Deserialize)]
struct Msg {
    #[serde(default)]
    model: Option<String>,
    #[serde(default, deserialize_with = "blocks")]
    content: Vec<Block>,
}

#[derive(Deserialize, Default)]
struct Block {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, deserialize_with = "tool_input")]
    input: Option<ToolInput>,
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default, deserialize_with = "truthy")]
    is_error: bool,
}

#[derive(Deserialize, Default)]
struct ToolInput {
    #[serde(default, deserialize_with = "short")]
    description: Option<String>,
    #[serde(default, deserialize_with = "short")]
    file_path: Option<String>,
    #[serde(default, deserialize_with = "short")]
    notebook_path: Option<String>,
    #[serde(default, deserialize_with = "short")]
    command: Option<String>,
    #[serde(default, deserialize_with = "short")]
    pattern: Option<String>,
    #[serde(default, deserialize_with = "short")]
    path: Option<String>,
    #[serde(default, deserialize_with = "short")]
    query: Option<String>,
    #[serde(default, deserialize_with = "short")]
    url: Option<String>,
    #[serde(default, deserialize_with = "short")]
    skill: Option<String>,
    #[serde(default, deserialize_with = "short")]
    to: Option<String>,
    #[serde(default, deserialize_with = "short")]
    summary: Option<String>,
}

/// `content` is either a string (ignored) or an array of blocks.
fn blocks<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Block>, D::Error> {
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Vec<Block>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("message content")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<Block>, A::Error> {
            let mut out = Vec::new();
            while let Some(b) = seq.next_element::<Block>()? {
                out.push(b);
            }
            Ok(out)
        }
        fn visit_str<E: de::Error>(self, _: &str) -> Result<Vec<Block>, E> {
            Ok(Vec::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<Vec<Block>, E> {
            Ok(Vec::new())
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vec<Block>, A::Error> {
            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            Ok(Vec::new())
        }
    }
    d.deserialize_any(V)
}

/// A string field cut to 400 characters; any other JSON type becomes `None`.
fn short<'de, D: Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Option<String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("any value")
        }
        fn visit_str<E: de::Error>(self, s: &str) -> Result<Option<String>, E> {
            Ok(Some(s.chars().take(400).collect()))
        }
        fn visit_unit<E: de::Error>(self) -> Result<Option<String>, E> {
            Ok(None)
        }
        fn visit_bool<E: de::Error>(self, _: bool) -> Result<Option<String>, E> {
            Ok(None)
        }
        fn visit_i64<E: de::Error>(self, _: i64) -> Result<Option<String>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, _: u64) -> Result<Option<String>, E> {
            Ok(None)
        }
        fn visit_f64<E: de::Error>(self, _: f64) -> Result<Option<String>, E> {
            Ok(None)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Option<String>, A::Error> {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Option<String>, A::Error> {
            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            Ok(None)
        }
    }
    d.deserialize_any(V)
}

fn tool_input<'de, D: Deserializer<'de>>(d: D) -> Result<Option<ToolInput>, D::Error> {
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Option<ToolInput>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("tool input")
        }
        fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Option<ToolInput>, A::Error> {
            ToolInput::deserialize(de::value::MapAccessDeserializer::new(map)).map(Some)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Option<ToolInput>, E> {
            Ok(None)
        }
        fn visit_str<E: de::Error>(self, _: &str) -> Result<Option<ToolInput>, E> {
            Ok(None)
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Option<ToolInput>, A::Error> {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
            Ok(None)
        }
    }
    d.deserialize_any(V)
}

fn truthy<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = bool;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a flag")
        }
        fn visit_bool<E: de::Error>(self, b: bool) -> Result<bool, E> {
            Ok(b)
        }
        fn visit_unit<E: de::Error>(self) -> Result<bool, E> {
            Ok(false)
        }
        fn visit_str<E: de::Error>(self, s: &str) -> Result<bool, E> {
            Ok(s == "true")
        }
    }
    d.deserialize_any(V)
}

#[derive(Deserialize, Default)]
struct Meta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

/// read: looking something up. write: changing a file. run: a shell command.
/// talk: delegating to or messaging another agent. other: anything else.
fn classify(tool: &str) -> &'static str {
    match tool {
        "Read" | "Grep" | "Glob" | "LS" | "NotebookRead" | "ToolSearch" | "WebFetch" | "WebSearch" => "read",
        "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => "write",
        "Bash" | "BashOutput" | "KillShell" | "KillBash" | "Monitor" => "run",
        "Agent" | "Task" | "SendMessage" | "Skill" | "TaskStop" | "TaskCreate" | "TaskUpdate" | "TodoWrite" | "AskUserQuestion" => "talk",
        t if t.starts_with("mcp__") => {
            let t = t.to_ascii_lowercase();
            if ["search", "scrape", "fetch", "read", "explore", "query", "docs", "get_", "list_", "find"].iter().any(|k| t.contains(k)) {
                "read"
            } else {
                "other"
            }
        }
        _ => "other",
    }
}

fn display_tool(tool: &str) -> String {
    match tool.strip_prefix("mcp__") {
        Some(rest) => match rest.rsplit_once("__") {
            Some((server, name)) => format!("{name} ({})", server.replace("claude_ai_", "")),
            None => rest.to_string(),
        },
        None => tool.to_string(),
    }
}

#[derive(Clone, Serialize)]
pub struct Call {
    pub t_ms: i64,
    pub end_ms: Option<i64>,
    pub tool: String,
    pub kind: &'static str,
    /// The tool's own description, or the file path, or a short command summary.
    pub label: String,
    /// A short command summary when the label is a description.
    pub detail: String,
    pub error: bool,
    #[serde(skip)]
    path: Option<String>,
    #[serde(skip)]
    id: String,
}

#[derive(Clone, Serialize, Default)]
pub struct FileTouch {
    pub path: String,
    pub reads: u32,
    pub writes: u32,
    pub last_ms: i64,
}

struct Worker {
    file: PathBuf,
    offset: u64,
    partial: Vec<u8>,
    session: String,
    is_lead: bool,
    name: String,
    description: String,
    model: Option<String>,
    first_ms: Option<i64>,
    last_ms: Option<i64>,
    calls: Vec<Call>,
    open: HashMap<String, usize>,
    files: HashMap<String, FileTouch>,
    errors: u64,
    bad_lines: u64,
    bytes: u64,
}


#[derive(Serialize)]
pub struct WorkerView {
    pub id: String,
    pub session: String,
    pub is_lead: bool,
    pub name: String,
    pub description: String,
    pub model: Option<String>,
    /// working | idle | quiet
    pub state: &'static str,
    pub first_ms: Option<i64>,
    pub last_ms: Option<i64>,
    pub current: Option<Call>,
    pub last_call: Option<Call>,
    pub calls_total: usize,
    pub calls_in_window: usize,
    pub errors_total: u64,
    pub kinds: [usize; 5],
    pub files: Vec<FileTouch>,
    pub files_total: usize,
    /// One entry per time bucket: calls by kind [read, write, run, talk, other] and errors.
    pub buckets: Vec<[u16; 6]>,
    /// Up to four call labels for each bucket that has calls, keyed by bucket index.
    pub samples: BTreeMap<usize, Vec<String>>,
    pub recent: Vec<Call>,
    pub bad_lines: u64,
    pub bytes: u64,
    pub file: String,
}

#[derive(Serialize)]
pub struct WorkersView {
    pub window_s: u64,
    pub bucket_s: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub workers: Vec<WorkerView>,
    pub dir: String,
    pub dir_exists: bool,
    pub older_sessions_hidden: usize,
    pub scanned_ms: Option<i64>,
}

pub struct Store {
    workers: HashMap<PathBuf, Worker>,
    pub dir_exists: bool,
    pub older_hidden: usize,
    pub scanned_ms: Option<i64>,
}

pub const BUCKETS: usize = 180;

impl Store {
    pub fn new() -> Store {
        Store { workers: HashMap::new(), dir_exists: false, older_hidden: 0, scanned_ms: None }
    }

    /// Finds transcript files and reads whatever was appended since the last scan.
    /// Lead sessions whose file was last written before the study started are left out.
    pub fn scan(&mut self, cfg: &Config, study_start_ms: i64) {
        let dir = &cfg.transcripts_dir;
        self.dir_exists = dir.is_dir();
        self.scanned_ms = Some(cfg.now_ms());
        let mut found: Vec<(PathBuf, String, bool)> = Vec::new();
        let mut older = 0;
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(session) = path.file_stem().and_then(|s| s.to_str()).map(String::from) else { continue };
            let recent = cfg.demo || util::file_mtime_ms(&path).is_some_and(|m| m >= study_start_ms);
            if !recent {
                older += 1;
                continue;
            }
            let subagents = dir.join(&session).join("subagents");
            for sub in std::fs::read_dir(&subagents).into_iter().flatten().flatten() {
                let sp = sub.path();
                let is_agent = sp.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("agent-") && n.ends_with(".jsonl"));
                if is_agent {
                    found.push((sp, session.clone(), false));
                }
            }
            found.push((path, session, true));
        }
        self.older_hidden = older;
        self.workers.retain(|path, _| found.iter().any(|(p, _, _)| p == path));
        for (path, session, is_lead) in found {
            let worker = self.workers.entry(path.clone()).or_insert_with(|| new_worker(&path, session, is_lead));
            read_appended(worker, cfg);
        }
    }

    pub fn view(&self, cfg: &Config, window_s: u64) -> WorkersView {
        let now = cfg.now_ms();
        let window_s = window_s.clamp(300, 7 * 86400);
        let start = now - window_s as i64 * 1000;
        let bucket_ms = window_s as f64 * 1000.0 / BUCKETS as f64;
        let mut workers: Vec<WorkerView> = self.workers.values().map(|w| worker_view(w, cfg, now, start, bucket_ms)).collect();
        // Leads first, then by name; a stable order so rows do not jump while reading.
        workers.sort_by(|a, b| b.is_lead.cmp(&a.is_lead).then(a.first_ms.cmp(&b.first_ms)).then(a.name.cmp(&b.name)));
        WorkersView {
            window_s,
            bucket_s: bucket_ms / 1000.0,
            start_ms: start,
            end_ms: now,
            workers,
            dir: cfg.short_path(&cfg.transcripts_dir.to_string_lossy()),
            dir_exists: self.dir_exists,
            older_sessions_hidden: self.older_hidden,
            scanned_ms: self.scanned_ms,
        }
    }
}

fn new_worker(path: &Path, session: String, is_lead: bool) -> Worker {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let mut name = if is_lead { "Lead session".to_string() } else { stem.trim_start_matches("agent-").to_string() };
    let mut description = if is_lead { format!("Lead session {}", &session[..session.len().min(8)]) } else { String::new() };
    let mut model = None;
    if !is_lead {
        let meta_path = path.with_extension("meta.json");
        if let Some(meta) = std::fs::read_to_string(meta_path).ok().and_then(|t| serde_json::from_str::<Meta>(&t).ok()) {
            if let Some(n) = meta.name {
                name = n;
            }
            description = meta.description.unwrap_or_default();
            model = meta.model;
        }
    }
    Worker {
        file: path.to_path_buf(),
        offset: 0,
        partial: Vec::new(),
        session,
        is_lead,
        name,
        description,
        model,
        first_ms: None,
        last_ms: None,
        calls: Vec::new(),
        open: HashMap::new(),
        files: HashMap::new(),
        errors: 0,
        bad_lines: 0,
        bytes: 0,
    }
}

fn read_appended(w: &mut Worker, cfg: &Config) {
    let Ok(mut file) = std::fs::File::open(&w.file) else { return };
    let Ok(len) = file.metadata().map(|m| m.len()) else { return };
    if len < w.offset {
        // The file was replaced by a shorter one: start over.
        let fresh = new_worker(&w.file, w.session.clone(), w.is_lead);
        *w = fresh;
    }
    if len == w.offset {
        return;
    }
    if file.seek(SeekFrom::Start(w.offset)).is_err() {
        return;
    }
    let mut buf = std::mem::take(&mut w.partial);
    if (&mut file).take(len - w.offset).read_to_end(&mut buf).is_err() {
        w.partial = buf;
        return;
    }
    w.offset = len;
    w.bytes = len;
    let complete = buf.iter().rposition(|b| *b == b'\n').map(|i| i + 1).unwrap_or(0);
    for line in buf[..complete].split(|b| *b == b'\n') {
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        match serde_json::from_slice::<Line>(line) {
            Ok(parsed) => apply(w, parsed, cfg),
            Err(_) => w.bad_lines += 1,
        }
    }
    w.partial = buf[complete..].to_vec();
}

fn apply(w: &mut Worker, line: Line, cfg: &Config) {
    let kind = line.kind.as_deref().unwrap_or("");
    if kind == "ai-title" {
        if let Some(title) = line.ai_title.filter(|t| !t.trim().is_empty()) {
            if w.is_lead {
                w.name = util::one_line(&title, 60);
            }
        }
        return;
    }
    if kind != "assistant" && kind != "user" {
        return;
    }
    let Some(t_ms) = line.timestamp.as_deref().and_then(util::parse_iso_ms) else { return };
    let Some(msg) = line.message else { return };
    w.first_ms.get_or_insert(t_ms);
    w.last_ms = Some(w.last_ms.map_or(t_ms, |l| l.max(t_ms)));
    if kind == "assistant" {
        if let Some(model) = msg.model.filter(|m| !m.starts_with('<')) {
            w.model = Some(model);
        }
    }
    for block in msg.content {
        match (kind, block.kind.as_str()) {
            ("assistant", "tool_use") => {
                let tool = block.name.unwrap_or_else(|| "tool".into());
                let input = block.input.unwrap_or_default();
                let call = describe(&tool, input, t_ms, block.id.unwrap_or_default(), cfg);
                if let Some(path) = &call.path {
                    let touch = w.files.entry(path.clone()).or_insert_with(|| FileTouch { path: path.clone(), ..Default::default() });
                    if call.kind == "write" {
                        touch.writes += 1;
                    } else {
                        touch.reads += 1;
                    }
                    touch.last_ms = t_ms;
                }
                if !call.id.is_empty() {
                    w.open.insert(call.id.clone(), w.calls.len());
                }
                w.calls.push(call);
            }
            ("user", "tool_result") => {
                let Some(index) = block.tool_use_id.and_then(|id| w.open.remove(&id)) else { continue };
                if let Some(call) = w.calls.get_mut(index) {
                    call.end_ms = Some(t_ms);
                    call.error = block.is_error;
                    if block.is_error {
                        w.errors += 1;
                    }
                }
            }
            _ => {}
        }
    }
}

fn describe(tool: &str, input: ToolInput, t_ms: i64, id: String, cfg: &Config) -> Call {
    let kind = classify(tool);
    let file = input.file_path.or(input.notebook_path).map(|p| cfg.short_path(&p));
    let command = input.command.as_deref().map(|c| util::one_line(c, 140)).unwrap_or_default();
    let (label, detail) = if let Some(path) = &file {
        (path.clone(), String::new())
    } else if tool == "Bash" {
        match input.description {
            Some(d) => (util::one_line(&d, 120), command),
            None => (command, String::new()),
        }
    } else if tool == "SendMessage" {
        (
            util::one_line(&format!("to {}: {}", input.to.unwrap_or_default(), input.summary.unwrap_or_default()), 120),
            String::new(),
        )
    } else if let Some(pattern) = input.pattern {
        let scope = input.path.map(|p| format!(" in {}", cfg.short_path(&p))).unwrap_or_default();
        (util::one_line(&format!("{pattern}{scope}"), 120), String::new())
    } else {
        let text = input.description.or(input.skill).or(input.query).or(input.url).or(input.summary).unwrap_or_default();
        (util::one_line(&text, 120), command)
    };
    Call { t_ms, end_ms: None, tool: display_tool(tool), kind, label, detail, error: false, path: file, id }
}

fn worker_view(w: &Worker, cfg: &Config, now: i64, start: i64, bucket_ms: f64) -> WorkerView {
    let mut buckets = vec![[0u16; 6]; BUCKETS];
    let mut samples: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let from = w.calls.partition_point(|c| c.t_ms < start);
    let mut kinds = [0usize; 5];
    for call in &w.calls[from..] {
        let i = (((call.t_ms - start) as f64 / bucket_ms) as usize).min(BUCKETS - 1);
        let k = match call.kind {
            "read" => 0,
            "write" => 1,
            "run" => 2,
            "talk" => 3,
            _ => 4,
        };
        buckets[i][k] = buckets[i][k].saturating_add(1);
        kinds[k] += 1;
        if call.error {
            buckets[i][5] = buckets[i][5].saturating_add(1);
        }
        let list = samples.entry(i).or_default();
        if list.len() < 4 {
            list.push(format!("{}: {}", call.tool, call.label));
        }
    }
    // A tool call with no result yet is in flight, unless it is so old that the session
    // must have ended without one.
    let current = w
        .calls
        .iter()
        .rev()
        .take(12)
        .find(|c| c.end_ms.is_none() && now - c.t_ms < 30 * 60_000)
        .cloned();
    let age = w.last_ms.map(|l| now - l);
    let state = match (age, &current) {
        (Some(a), _) if a < 120_000 => "working",
        (Some(a), Some(_)) if a < 15 * 60_000 => "working",
        (Some(a), _) if a < 20 * 60_000 => "idle",
        _ => "quiet",
    };
    let mut files: Vec<FileTouch> = w.files.values().cloned().collect();
    files.sort_by(|a, b| b.last_ms.cmp(&a.last_ms));
    let files_total = files.len();
    files.truncate(14);
    WorkerView {
        id: w.file.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string(),
        session: w.session.clone(),
        is_lead: w.is_lead,
        name: w.name.clone(),
        description: w.description.clone(),
        model: w.model.clone(),
        state,
        first_ms: w.first_ms,
        last_ms: w.last_ms,
        current: if state == "working" { current } else { None },
        last_call: w.calls.last().cloned(),
        calls_total: w.calls.len(),
        calls_in_window: w.calls.len() - from,
        errors_total: w.errors,
        kinds,
        files,
        files_total,
        buckets,
        samples,
        recent: w.calls.iter().rev().take(24).cloned().collect(),
        bad_lines: w.bad_lines,
        bytes: w.bytes,
        file: cfg.short_path(&w.file.to_string_lossy()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_call_line_yields_name_label_and_no_output() {
        let line = r#"{"type":"assistant","timestamp":"2026-11-13T07:55:13.892Z","message":{"model":"claude-x","content":[{"type":"thinking","thinking":"long"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls -la\nsecond line","description":"List files","timeout":5}}]},"wireToolInputs":{"t1":{"command":"x"}}}"#;
        let parsed: Line = serde_json::from_str(line).unwrap();
        let msg = parsed.message.unwrap();
        assert_eq!(msg.content.len(), 2);
        let input = msg.content.into_iter().nth(1).unwrap().input.unwrap();
        assert_eq!(input.description.as_deref(), Some("List files"));
    }

    #[test]
    fn a_result_line_keeps_only_id_and_error_flag() {
        let line = r#"{"type":"user","timestamp":"2026-11-13T07:55:14Z","message":{"role":"user","content":[{"tool_use_id":"t1","type":"tool_result","content":[{"type":"text","text":"SECRET OUTPUT"}],"is_error":true}]},"toolUseResult":{"stdout":"SECRET OUTPUT"}}"#;
        let parsed: Line = serde_json::from_str(line).unwrap();
        let block = parsed.message.unwrap().content.into_iter().next().unwrap();
        assert_eq!(block.tool_use_id.as_deref(), Some("t1"));
        assert!(block.is_error);
    }

    #[test]
    fn string_content_and_odd_types_do_not_fail_the_line() {
        let a: Line = serde_json::from_str(r#"{"type":"user","message":{"content":"plain text"}}"#).unwrap();
        assert!(a.message.unwrap().content.is_empty());
        let b: Line = serde_json::from_str(r#"{"type":"assistant","message":{"content":[{"type":"tool_use","name":"X","input":{"description":42,"file_path":["a"]}}]}}"#).unwrap();
        let input = b.message.unwrap().content.into_iter().next().unwrap().input.unwrap();
        assert!(input.description.is_none() && input.file_path.is_none());
    }

    #[test]
    fn tools_are_grouped_by_what_they_do() {
        assert_eq!(classify("Read"), "read");
        assert_eq!(classify("Edit"), "write");
        assert_eq!(classify("Bash"), "run");
        assert_eq!(classify("SendMessage"), "talk");
        assert_eq!(classify("mcp__codegraph__codegraph_explore"), "read");
        assert_eq!(display_tool("mcp__claude_ai_Firecrawl_Pi__firecrawl_scrape"), "firecrawl_scrape (Firecrawl_Pi)");
    }
}
