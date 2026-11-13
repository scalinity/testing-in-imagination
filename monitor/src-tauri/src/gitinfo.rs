//! Commits and uncommitted work on the task branch. Read-only git commands only, with
//! `GIT_OPTIONAL_LOCKS=0` so that not even the index is refreshed on disk.

use crate::config::{Config, BRANCH, STUDY_START_COMMIT};
use crate::util;
use serde::Serialize;
use std::time::Duration;

pub const AREAS: [&str; 4] = ["tii-cee/", "reports/research-advancement/", "monitor/", "other"];

fn area_of(path: &str) -> usize {
    AREAS[..3].iter().position(|a| path.starts_with(a)).unwrap_or(3)
}

#[derive(Serialize, Clone, Default)]
pub struct Delta {
    pub ins: u64,
    pub del: u64,
    pub files: u32,
}

#[derive(Serialize, Clone)]
pub struct Commit {
    pub hash: String,
    pub t_ms: i64,
    pub subject: String,
    pub areas: [Delta; 4],
}

#[derive(Serialize, Clone, Default)]
pub struct GitSnap {
    pub ok: bool,
    pub error: String,
    pub branch: String,
    pub head_branch: String,
    pub start_commit: String,
    pub areas: [&'static str; 4],
    pub commits: Vec<Commit>,
    /// Changes to tracked files against HEAD.
    pub modified: [Delta; 4],
    /// Files git does not track yet (ignored files excluded); `ins` holds their line count.
    pub untracked: [Delta; 4],
}

fn git(cfg: &Config, args: &[&str]) -> Result<String, String> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    match util::run("/usr/bin/git", &args, &[("GIT_OPTIONAL_LOCKS", "0")], Some(&cfg.repo), Duration::from_secs(20), 32 << 20) {
        Ok(out) if out.ok() => Ok(out.text()),
        Ok(out) if out.timed_out => Err(format!("git {} timed out", args.first().cloned().unwrap_or_default())),
        Ok(out) => Err(util::one_line(&out.stderr, 200)),
        Err(e) => Err(format!("could not run git: {e}")),
    }
}

/// Parses `git log --numstat --format=%x1e%H%x1f%at%x1f%s`, newest first, into oldest first.
pub fn parse_log(text: &str) -> Vec<Commit> {
    let mut commits = Vec::new();
    for record in text.split('\u{1e}').filter(|r| !r.trim().is_empty()) {
        let mut lines = record.lines();
        let Some(head) = lines.next() else { continue };
        let mut parts = head.splitn(3, '\u{1f}');
        let (Some(hash), Some(time), Some(subject)) = (parts.next(), parts.next(), parts.next()) else { continue };
        let Ok(secs) = time.trim().parse::<i64>() else { continue };
        let mut areas: [Delta; 4] = Default::default();
        add_numstat(lines, &mut areas);
        commits.push(Commit { hash: hash.trim().to_string(), t_ms: secs * 1000, subject: util::one_line(subject, 160), areas });
    }
    commits.reverse();
    commits
}

fn add_numstat<'a>(lines: impl Iterator<Item = &'a str>, areas: &mut [Delta; 4]) {
    for line in lines {
        let mut cols = line.splitn(3, '\t');
        let (Some(ins), Some(del), Some(path)) = (cols.next(), cols.next(), cols.next()) else { continue };
        let slot = &mut areas[area_of(path)];
        slot.ins += ins.parse::<u64>().unwrap_or(0); // "-" marks a binary file
        slot.del += del.parse::<u64>().unwrap_or(0);
        slot.files += 1;
    }
}

pub fn read(cfg: &Config) -> GitSnap {
    let mut snap = GitSnap {
        branch: BRANCH.to_string(),
        start_commit: STUDY_START_COMMIT.to_string(),
        areas: AREAS,
        ..Default::default()
    };
    if cfg.demo {
        let dir = cfg.fixtures.join("mac/git");
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        snap.commits = parse_log(&read("log.txt"));
        add_numstat(read("diff-numstat.txt").lines(), &mut snap.modified);
        add_numstat(read("untracked-numstat.txt").lines(), &mut snap.untracked);
        snap.head_branch = BRANCH.to_string();
        snap.ok = !snap.commits.is_empty();
        if !snap.ok {
            snap.error = "fixtures/mac/git/log.txt is missing or empty".into();
        }
        return snap;
    }
    let range = format!("{STUDY_START_COMMIT}^..{BRANCH}");
    match git(cfg, &["log", "--numstat", "--no-renames", "--format=%x1e%H%x1f%at%x1f%s", &range, "--"]) {
        Ok(text) => {
            snap.commits = parse_log(&text);
            snap.ok = true;
        }
        Err(e) => {
            snap.error = e;
            return snap;
        }
    }
    snap.head_branch = git(cfg, &["rev-parse", "--abbrev-ref", "HEAD"]).map(|s| s.trim().to_string()).unwrap_or_default();
    if let Ok(text) = git(cfg, &["diff", "--numstat", "--no-renames", "HEAD", "--"]) {
        add_numstat(text.lines(), &mut snap.modified);
    }
    if let Ok(text) = git(cfg, &["ls-files", "--others", "--exclude-standard", "-z"]) {
        for path in text.split('\0').filter(|p| !p.is_empty()).take(5000) {
            let slot = &mut snap.untracked[area_of(path)];
            slot.files += 1;
            slot.ins += count_lines(&cfg.repo.join(path));
        }
    }
    snap
}

/// Lines in a text file of at most 2 MB; 0 for anything larger or binary.
fn count_lines(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else { return 0 };
    if !meta.is_file() || meta.len() > 2 << 20 {
        return 0;
    }
    let Ok(bytes) = std::fs::read(path) else { return 0 };
    if bytes.iter().take(8192).any(|b| *b == 0) {
        return 0;
    }
    bytes.iter().filter(|b| **b == b'\n').count() as u64
}

/// Author time of the commit that started the study, used to leave out older sessions.
pub fn study_start_ms(cfg: &Config) -> Option<i64> {
    git(cfg, &["show", "-s", "--format=%at", STUDY_START_COMMIT, "--"])
        .ok()
        .and_then(|t| t.trim().parse::<i64>().ok())
        .map(|s| s * 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_records_are_split_by_area() {
        let text = "\u{1e}bbb\u{1f}200\u{1f}second\n\n3\t1\ttii-cee/a.py\n-\t-\tmonitor/icon.png\n\u{1e}aaa\u{1f}100\u{1f}first\n\n10\t0\treports/research-advancement/SPEC.md\n2\t2\tREADME.md\n";
        let commits = parse_log(text);
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].hash, "aaa");
        assert_eq!(commits[0].areas[1].ins, 10);
        assert_eq!(commits[0].areas[3].del, 2);
        assert_eq!(commits[1].areas[0].ins, 3);
        assert_eq!(commits[1].areas[2].files, 1);
        assert_eq!(commits[1].areas[2].ins, 0);
    }
}
