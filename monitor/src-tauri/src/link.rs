//! Connection health on the Mac: the keeper's log, the SSH master, and the launchd jobs.

use crate::config::{Config, LAUNCHD_LABELS};
use crate::{remote, util};
use serde::Serialize;
use std::time::Duration;

#[derive(Serialize, Clone)]
pub struct KeeperEntry {
    pub t_ms: i64,
    /// ok | reconnected | relayed | no_answer | closed | signin | down | other
    pub kind: &'static str,
    pub text: String,
}

#[derive(Serialize, Clone, Default)]
pub struct LaunchdJob {
    pub label: String,
    pub installed: bool,
    /// launchd's own word: running, not running, ...
    pub state: String,
    pub pid: Option<u32>,
    pub last_exit: Option<i32>,
    pub runs: Option<u64>,
    pub interval_s: Option<u64>,
    pub note: String,
}

#[derive(Serialize, Clone, Default)]
pub struct LinkSnap {
    pub master_up: bool,
    pub master_detail: String,
    pub keeper: Vec<KeeperEntry>,
    pub keeper_log: String,
    pub keeper_log_exists: bool,
    pub keeper_log_mtime_ms: Option<i64>,
    pub keeper_unparsed: u32,
    pub launchd: Vec<LaunchdJob>,
}

pub fn classify_keeper(text: &str) -> &'static str {
    let t = text.trim();
    if t == "ok" {
        "ok"
    } else if t == "reconnected" {
        "reconnected"
    } else if t.starts_with("relayed") {
        "relayed"
    } else if t.starts_with("master present but did not answer") {
        "no_answer"
    } else if t.starts_with("closed unresponsive master") {
        "closed"
    } else if t.starts_with("notified: sign-in needed") {
        "signin"
    } else if t.starts_with("down") {
        "down"
    } else {
        "other"
    }
}

pub fn parse_keeper_log(text: &str) -> (Vec<KeeperEntry>, u32) {
    let mut entries = Vec::new();
    let mut unparsed = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (stamp, rest) = line.split_once(' ').unwrap_or((line, ""));
        match util::parse_iso_ms(stamp) {
            Some(t_ms) => entries.push(KeeperEntry { t_ms, kind: classify_keeper(rest), text: util::one_line(rest, 120) }),
            None => unparsed += 1,
        }
    }
    (entries, unparsed)
}

/// Reads the first-level fields of `launchctl print` output (one leading tab).
pub fn parse_launchctl(label: &str, text: &str) -> LaunchdJob {
    let mut job = LaunchdJob { label: label.to_string(), installed: true, ..Default::default() };
    for line in text.lines() {
        let Some(body) = line.strip_prefix('\t') else { continue };
        if body.starts_with('\t') {
            continue;
        }
        let Some((key, value)) = body.split_once(" = ") else { continue };
        let value = value.trim();
        match key.trim() {
            "state" => job.state = value.to_string(),
            "pid" => job.pid = value.parse().ok(),
            "last exit code" => job.last_exit = value.split_whitespace().next().and_then(|v| v.parse().ok()),
            "runs" => job.runs = value.parse().ok(),
            "run interval" => job.interval_s = value.split_whitespace().next().and_then(|v| v.parse().ok()),
            _ => {}
        }
    }
    job
}

pub fn read_keeper(cfg: &Config, snap: &mut LinkSnap) {
    snap.keeper_log = cfg.short_path(&cfg.keeper_log.to_string_lossy());
    match std::fs::read_to_string(&cfg.keeper_log) {
        Ok(text) => {
            let (mut entries, unparsed) = parse_keeper_log(&text);
            let extra = entries.len().saturating_sub(3000);
            entries.drain(..extra);
            snap.keeper = entries;
            snap.keeper_unparsed = unparsed;
            snap.keeper_log_exists = true;
            snap.keeper_log_mtime_ms = util::file_mtime_ms(&cfg.keeper_log);
        }
        Err(_) => {
            snap.keeper.clear();
            snap.keeper_log_exists = false;
            snap.keeper_log_mtime_ms = None;
        }
    }
}

pub fn read_master(cfg: &Config, snap: &mut LinkSnap) -> (bool, String) {
    let (up, detail) = if cfg.demo {
        let text = std::fs::read_to_string(cfg.fixtures.join("mac/ssh-check.txt")).unwrap_or_default();
        (text.contains("Master running"), util::one_line(&text, 160))
    } else {
        remote::master_check()
    };
    snap.master_up = up;
    snap.master_detail = detail.clone();
    (up, detail)
}

pub fn read_launchd(cfg: &Config, snap: &mut LinkSnap) {
    let mut jobs = Vec::new();
    for label in LAUNCHD_LABELS {
        if cfg.demo {
            let path = cfg.fixtures.join(format!("mac/launchctl/{label}.txt"));
            jobs.push(match std::fs::read_to_string(path) {
                Ok(text) => parse_launchctl(label, &text),
                Err(_) => LaunchdJob { label: label.to_string(), note: "not installed".into(), ..Default::default() },
            });
            continue;
        }
        let uid = unsafe { getuid() };
        let args = ["print".to_string(), format!("gui/{uid}/{label}")];
        jobs.push(match util::run("/bin/launchctl", &args, &[], None, Duration::from_secs(5), 65536) {
            Ok(out) if out.ok() => parse_launchctl(label, &out.text()),
            Ok(out) => LaunchdJob {
                label: label.to_string(),
                note: if out.stderr.contains("Could not find service") || out.text().contains("Could not find service") {
                    "not installed".into()
                } else {
                    util::one_line(&format!("{} {}", out.text(), out.stderr), 140)
                },
                ..Default::default()
            },
            Err(e) => LaunchdJob { label: label.to_string(), note: format!("could not run launchctl: {e}"), ..Default::default() },
        });
    }
    snap.launchd = jobs;
}

extern "C" {
    fn getuid() -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeper_lines_are_classified() {
        let log = "2026-11-13T02:45:18-0500 ok\n2026-11-13T02:50:22-0500 notified: sign-in needed\n2026-11-13T02:55:26-0500 down, already notified\n2026-11-13T03:00:29-0500 reconnected\n2026-11-13T03:05:29-0500 relayed 2 event(s)\ngarbage\n";
        let (entries, unparsed) = parse_keeper_log(log);
        let kinds: Vec<&str> = entries.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, ["ok", "signin", "down", "reconnected", "relayed"]);
        assert_eq!(unparsed, 1);
        assert_eq!(entries[1].t_ms - entries[0].t_ms, 304_000);
    }

    #[test]
    fn launchctl_first_level_fields_only() {
        let text = "gui/501/com.tii.hpg-keeper = {\n\tactive count = 0\n\tstate = not running\n\n\truns = 8\n\tlast exit code = 0\n\n\tresource coalition = {\n\t\tstate = active\n\t}\n\n\trun interval = 300 seconds\n}\n";
        let job = parse_launchctl("com.tii.hpg-keeper", text);
        assert_eq!(job.state, "not running");
        assert_eq!(job.runs, Some(8));
        assert_eq!(job.last_exit, Some(0));
        assert_eq!(job.interval_s, Some(300));
    }
}
