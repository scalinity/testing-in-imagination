//! Small shared helpers: time parsing, bounded process execution, markdown tables.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Parses an ISO 8601 time with an offset. Accepts both `-05:00` and the keeper log's `-0500`.
pub fn parse_iso_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    chrono::DateTime::parse_from_rfc3339(s)
        .or_else(|_| chrono::DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%z"))
        .ok()
        .map(|d| d.timestamp_millis())
}

pub fn real_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn mtime_ms(meta: &std::fs::Metadata) -> Option<i64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
}

pub fn file_mtime_ms(path: &std::path::Path) -> Option<i64> {
    std::fs::metadata(path).ok().and_then(|m| mtime_ms(&m))
}

/// Cuts a string to at most `max` characters on a character boundary, on one line.
pub fn one_line(s: &str, max: usize) -> String {
    let mut out = String::with_capacity(max.min(s.len()) + 1);
    let mut last_space = false;
    for ch in s.chars() {
        let ch = if ch.is_whitespace() { ' ' } else { ch };
        if ch == ' ' && last_space {
            continue;
        }
        last_space = ch == ' ';
        if out.chars().count() >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out.trim().to_string()
}

pub struct Out {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub timed_out: bool,
    pub ms: u64,
}

impl Out {
    pub fn ok(&self) -> bool {
        self.code == Some(0) && !self.timed_out
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }
}

/// Runs a program with no stdin, a hard timeout and a cap on captured stdout.
pub fn run(
    program: &str,
    args: &[String],
    envs: &[(&str, &str)],
    cwd: Option<&std::path::Path>,
    timeout: Duration,
    max_stdout: usize,
) -> std::io::Result<Out> {
    let started = Instant::now();
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = cmd.spawn()?;
    let mut so = child.stdout.take().expect("piped stdout");
    let mut se = child.stderr.take().expect("piped stderr");
    let t_out = std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 65536];
        loop {
            match so.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if kept.len() < max_stdout {
                        let room = max_stdout - kept.len();
                        kept.extend_from_slice(&buf[..n.min(room)]);
                    }
                }
            }
        }
        kept
    });
    let t_err = std::thread::spawn(move || {
        let mut kept = Vec::new();
        let _ = (&mut se).take(16384).read_to_end(&mut kept);
        let mut sink = [0u8; 4096];
        while let Ok(n) = se.read(&mut sink) {
            if n == 0 {
                break;
            }
        }
        String::from_utf8_lossy(&kept).trim().to_string()
    });
    let mut timed_out = false;
    let code = loop {
        match child.try_wait()? {
            Some(status) => break status.code(),
            None => {
                if started.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        }
    };
    Ok(Out {
        code,
        stdout: t_out.join().unwrap_or_default(),
        stderr: t_err.join().unwrap_or_default(),
        timed_out,
        ms: started.elapsed().as_millis() as u64,
    })
}

/// Returns the lines of a markdown section: from the first line that starts with `heading`
/// up to the next heading of the same or a higher level.
pub fn md_section<'a>(text: &'a str, heading: &str) -> Vec<&'a str> {
    let level = heading.chars().take_while(|c| *c == '#').count();
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if !inside {
            if line.starts_with(heading) {
                inside = true;
            }
            continue;
        }
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && hashes <= level && line[hashes..].starts_with(' ') {
            break;
        }
        out.push(line);
    }
    out
}

/// Returns the body rows of the first markdown table found in `lines`.
pub fn md_table(lines: &[&str]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut seen = false;
    for line in lines {
        let t = line.trim();
        if t.starts_with('|') {
            seen = true;
            let cells: Vec<String> = t
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            let is_rule = cells
                .iter()
                .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'));
            if !is_rule {
                rows.push(cells);
            }
        } else if seen {
            break;
        }
    }
    if !rows.is_empty() {
        rows.remove(0); // header row
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_offset_styles() {
        let a = parse_iso_ms("2026-11-13T02:45:18-0500").unwrap();
        let b = parse_iso_ms("2026-11-13T02:45:18-05:00").unwrap();
        let c = parse_iso_ms("2026-11-13T07:45:18.000Z").unwrap();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert!(parse_iso_ms("not a time").is_none());
    }

    #[test]
    fn reads_a_markdown_table() {
        let text = "## 4. Things\n\nintro\n\n| ID | Status |\n|---|---|\n| E0 | PLANNED |\n| E1 | DONE |\n\n## 5. Next\n| x | y |\n";
        let rows = md_table(&md_section(text, "## 4."));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1], vec!["E1".to_string(), "DONE".to_string()]);
    }
}
