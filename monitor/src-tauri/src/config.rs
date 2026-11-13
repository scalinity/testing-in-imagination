//! Where every source lives, and which clock is in use.
//!
//! Real mode reads the real sources. Demo mode (`--demo`) replaces every source with the
//! fictional files in `monitor/fixtures/` and uses the fixtures' own clock, so nothing real
//! and nothing fictional ever share a screen.

use crate::util;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

pub const BRANCH: &str = "tii-cee-1/2026-11-13";
pub const STUDY_START_COMMIT: &str = "632bd18";
pub const LAUNCHD_LABELS: [&str; 3] = ["com.tii.hpg-keeper", "com.tii.caffeinate", "com.tii.local-loop"];
pub const DEFAULT_REMOTE_INTERVAL_S: u64 = 120;
/// Hard floor between two reads of HiPerGator. Not configurable.
pub const MIN_REMOTE_INTERVAL_S: u64 = 60;

pub struct Config {
    pub demo: bool,
    pub repo: PathBuf,
    /// Definitions that are the same in both modes: specification, authorization, cards.
    pub study_dir: PathBuf,
    /// Documents that change as the study runs: STATUS.md, BLOCKED.md, the ledger, gate records.
    pub live_docs_dir: PathBuf,
    pub figure_dir: PathBuf,
    pub transcripts_dir: PathBuf,
    pub keeper_log: PathBuf,
    pub fixtures: PathBuf,
    pub data_dir: PathBuf,
    /// The fixtures' clock, read from fixtures/demo.json on first use (never on the main thread).
    demo_anchor_ms: OnceLock<i64>,
    started: Instant,
}

impl Config {
    pub fn load(demo: bool) -> Config {
        // The repository is two levels above src-tauri/. Worked out from the path alone: nothing
        // under ~/Documents is touched here, because the first such access makes macOS ask for
        // permission and blocks until answered, and that must not happen before the window exists.
        let repo = std::env::var("TII_MONITOR_REPO").map(PathBuf::from).unwrap_or_else(|_| {
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest.parent().and_then(|p| p.parent()).map(PathBuf::from).unwrap_or(manifest)
        });
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
        let fixtures = repo.join("monitor/fixtures");
        let study_dir = repo.join("reports/research-advancement");
        let slug: String = repo
            .to_string_lossy()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        Config {
            demo,
            live_docs_dir: if demo { fixtures.join("mac/docs") } else { study_dir.clone() },
            figure_dir: if demo {
                fixtures.join("mac/figure1")
            } else {
                repo.join("tii-cee/results/figure1")
            },
            transcripts_dir: if demo {
                fixtures.join("mac/transcripts")
            } else {
                home.join(".claude/projects").join(slug)
            },
            keeper_log: if demo {
                fixtures.join("mac/keeper.log")
            } else {
                home.join("Library/Logs/tii-hpg-keeper.log")
            },
            data_dir: home.join("Library/Application Support/edu.ufl.tii-monitor"),
            study_dir,
            fixtures,
            repo,
            demo_anchor_ms: OnceLock::new(),
            started: Instant::now(),
        }
    }

    /// The clock every age on screen is measured against.
    pub fn now_ms(&self) -> i64 {
        if self.demo {
            let anchor = *self.demo_anchor_ms.get_or_init(|| {
                std::fs::read_to_string(self.fixtures.join("demo.json"))
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                    .and_then(|v| v.get("now").and_then(|n| n.as_str()).and_then(util::parse_iso_ms))
                    .unwrap_or_else(util::real_now_ms)
            });
            anchor + self.started.elapsed().as_millis() as i64
        } else {
            util::real_now_ms()
        }
    }

    /// Shortens an absolute path for display: relative to the repository, or with `~`.
    pub fn short_path(&self, p: &str) -> String {
        let repo = self.repo.to_string_lossy();
        if let Some(rest) = p.strip_prefix(repo.as_ref()) {
            let rest = rest.trim_start_matches('/');
            return if rest.is_empty() { ".".into() } else { rest.to_string() };
        }
        if let Ok(home) = std::env::var("HOME") {
            if let Some(rest) = p.strip_prefix(&home) {
                return format!("~{rest}");
            }
        }
        p.to_string()
    }
}
