//! The study's own documents, read from the repository: what every experiment and gate is,
//! what the plan's dates are, what is blocked. Hover text is built from these, so it stays
//! accurate when a document changes.

use crate::config::Config;
use crate::util::{self, md_section, md_table};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Serialize, Clone, Default)]
pub struct FileInfo {
    pub path: String,
    pub exists: bool,
    pub mtime_ms: Option<i64>,
}

#[derive(Serialize, Clone, Default)]
pub struct SpecExperiment {
    pub id: String,
    pub question: String,
    pub design: String,
    pub measurement: String,
    pub rule: String,
    pub kills: String,
    pub compute: String,
}

#[derive(Serialize, Clone, Default)]
pub struct SpecGate {
    pub id: String,
    pub rule: String,
    pub if_met: String,
    pub if_not: String,
}

#[derive(Serialize, Clone, Default)]
pub struct GpuRow {
    pub label: String,
    pub counted: String,
    pub estimate: Option<f64>,
    pub allowance: Option<f64>,
    pub basis: String,
    pub authorized: String,
}

#[derive(Serialize, Clone, Default)]
pub struct PlanRow {
    pub date: String,
    pub what: String,
}

#[derive(Serialize, Clone, Default)]
pub struct BlockedItem {
    pub id: String,
    pub title: String,
    pub opened: String,
    pub open: bool,
    /// An accepted gap: recorded, waits on nobody, never shown as a request.
    pub accepted: bool,
    pub status: String,
    pub waits: String,
    pub why: String,
    pub request: String,
    pub meanwhile: String,
}

#[derive(Serialize, Clone, Default)]
pub struct MilestoneState {
    pub id: String,
    pub state: String,
}

#[derive(Serialize, Clone, Default)]
pub struct RegisterRow {
    pub id: String,
    pub status: String,
    pub card: String,
    pub notes: String,
}

#[derive(Serialize, Clone, Default)]
pub struct Card {
    pub id: String,
    pub title: String,
    pub frozen: String,
    pub question: String,
    pub path: String,
    pub mtime_ms: Option<i64>,
}

/// One immutable gate record in GATE_DECISIONS/. A gate may be decided in parts, one file each
/// (`G1-twin.md`); a file named exactly after the gate (`G0.md`) decides it whole.
#[derive(Serialize, Clone, Default)]
pub struct GateRecord {
    pub name: String,
    pub mtime_ms: Option<i64>,
    pub title: String,
    pub evaluated: String,
    pub result: String,
    pub path: String,
}

fn first_paragraph(lines: &[&str]) -> String {
    let text: Vec<&str> = lines.iter().skip_while(|l| l.trim().is_empty()).take_while(|l| !l.trim().is_empty()).copied().collect();
    util::one_line(&text.join(" "), 420)
}

pub fn parse_gate_record(name: &str, text: &str, mtime_ms: Option<i64>) -> GateRecord {
    GateRecord {
        name: name.to_string(),
        mtime_ms,
        title: text.lines().next().unwrap_or("").trim_start_matches('#').trim().to_string(),
        evaluated: text.split("\n\n").find(|p| p.trim_start().starts_with("Evaluated")).map(|p| util::one_line(p, 260)).unwrap_or_default(),
        result: first_paragraph(&md_section(text, "## Result")),
        path: first_paragraph(&md_section(text, "## Path taken")),
    }
}

#[derive(Serialize, Clone, Default)]
pub struct FigureFile {
    pub name: String,
    pub path: String,
    pub exists: bool,
    pub mtime_ms: Option<i64>,
    pub json: Option<Value>,
    pub error: String,
}

#[derive(Serialize, Clone, Default, PartialEq, Debug)]
pub struct GpuLimit {
    pub account: String,
    pub gpus: u32,
}

#[derive(Serialize, Clone, Default)]
pub struct DocsSnap {
    pub spec: FileInfo,
    pub spec_version: String,
    pub spec_sha256: String,
    pub experiments: Vec<SpecExperiment>,
    pub gates: Vec<SpecGate>,
    pub gpu_estimates: Vec<GpuRow>,
    pub plan: Vec<PlanRow>,
    pub authorization: FileInfo,
    pub allowances: Vec<GpuRow>,
    pub addendum: FileInfo,
    /// GPUs allowed at once per account, from the addendum's "Compute added" table.
    pub gpu_limits: Vec<GpuLimit>,
    pub blocked_file: FileInfo,
    pub blocked: Vec<BlockedItem>,
    pub status_file: FileInfo,
    pub status_updated: String,
    pub status_milestone_line: String,
    pub milestones: Vec<MilestoneState>,
    pub ledger_file: FileInfo,
    pub register: Vec<RegisterRow>,
    pub cards: Vec<Card>,
    pub gate_records_dir: FileInfo,
    pub gate_records: Vec<GateRecord>,
    pub figure: Vec<FigureFile>,
    pub problems: Vec<String>,
}

fn info(cfg: &Config, path: &Path) -> FileInfo {
    FileInfo {
        path: cfg.short_path(&path.to_string_lossy()),
        exists: path.exists(),
        mtime_ms: util::file_mtime_ms(path),
    }
}

fn number(cell: &str) -> Option<f64> {
    let cleaned: String = cell.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
    if cleaned.is_empty() {
        None
    } else {
        cleaned.parse().ok()
    }
}

fn cell(row: &[String], i: usize) -> String {
    row.get(i).cloned().unwrap_or_default()
}

/// Splits a markdown list into bullets, joining continuation lines.
fn bullets(lines: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in lines {
        if let Some(rest) = line.strip_prefix("- ") {
            out.push(rest.trim().to_string());
        } else if line.starts_with(' ') || line.starts_with('>') || line.trim().is_empty() {
            if let Some(last) = out.last_mut() {
                let piece = line.trim().trim_start_matches('>').trim();
                if !piece.is_empty() {
                    last.push(' ');
                    last.push_str(piece);
                }
            }
        }
    }
    out
}

pub fn parse_blocked(text: &str) -> Vec<BlockedItem> {
    let lines: Vec<&str> = text.lines().collect();
    let starts: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.starts_with("## BLOCKED-")).map(|(i, _)| i).collect();
    // Items under a top-level "# Accepted gaps" heading are accepted gaps, not open requests.
    let accepted_from = lines.iter().position(|l| l.trim().eq_ignore_ascii_case("# Accepted gaps"));
    let mut items = Vec::new();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(lines.len());
        let head = lines[start].trim_start_matches("## ").trim();
        let (id, rest) = head.split_once(':').unwrap_or((head, ""));
        let (title, opened) = match rest.rfind("(opened ") {
            Some(i) => (rest[..i].trim(), rest[i + 8..].trim_end_matches(')').trim()),
            None => (rest.trim(), ""),
        };
        let in_accepted = accepted_from.map_or(false, |a| start > a);
        let mut item = BlockedItem { id: id.trim().to_string(), title: title.to_string(), opened: opened.to_string(), open: !in_accepted, accepted: in_accepted, ..Default::default() };
        for bullet in bullets(&lines[start + 1..end]) {
            let (key, value) = bullet.split_once(':').unwrap_or(("", &bullet));
            let value = value.trim().to_string();
            match key.trim() {
                "What waits" => item.waits = value,
                "Why it is blocked" => item.why = value,
                k if k.starts_with("Request") => item.request = value,
                "Meanwhile" => item.meanwhile = value,
                "Status" => {
                    let upper = value.to_ascii_uppercase();
                    item.accepted = item.accepted || upper.starts_with("ACCEPTED");
                    item.open = !(upper.starts_with("CLOSED") || item.accepted);
                    item.status = value;
                }
                _ => {}
            }
        }
        items.push(item);
    }
    items
}

/// Reads "M0 IN PROGRESS. M1 to M5 NOT STARTED." into one state per milestone.
pub fn parse_milestones(line: &str) -> Vec<MilestoneState> {
    const STATES: [&str; 4] = ["NOT STARTED", "IN PROGRESS", "COMPLETE", "BLOCKED"];
    let mut out: Vec<MilestoneState> = Vec::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut range_from: Option<u32> = None;
    let words: Vec<&str> = line.split(|c: char| c.is_whitespace() || c == ',' || c == ';').filter(|w| !w.is_empty()).collect();
    let mut i = 0;
    while i < words.len() {
        let w = words[i].trim_matches(|c: char| !c.is_ascii_alphanumeric());
        let milestone = w.strip_prefix('M').and_then(|d| d.parse::<u32>().ok()).filter(|d| *d <= 5);
        if let Some(m) = milestone {
            match range_from.take() {
                Some(from) => pending.extend(from + 1..=m),
                None => pending.push(m),
            }
        } else if w == "to" && !pending.is_empty() {
            range_from = pending.last().copied();
        } else {
            let rest = words[i..].join(" ");
            if let Some(state) = STATES.iter().find(|s| rest.starts_with(**s)) {
                for m in pending.drain(..) {
                    out.push(MilestoneState { id: format!("M{m}"), state: state.to_string() });
                }
                i += state.split(' ').count() - 1;
            }
        }
        i += 1;
    }
    out
}

fn read_spec(cfg: &Config, snap: &mut DocsSnap) {
    let path = cfg.study_dir.join("SPEC_RESEARCH_ADVANCEMENT.md");
    snap.spec = info(cfg, &path);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            snap.problems.push(format!("SPEC_RESEARCH_ADVANCEMENT.md could not be read: {e}."));
            return;
        }
    };
    snap.spec_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    let field = |name: &str| text.lines().take(20).find_map(|l| l.strip_prefix(name)).map(|v| v.trim().to_string()).unwrap_or_default();
    snap.spec_version = format!("{}, revised {}", field("Version:").split(" — ").next().unwrap_or("").trim(), field("Revision date:"));
    snap.experiments = md_table(&md_section(&text, "## 13. "))
        .iter()
        .filter(|r| r.len() >= 7)
        .map(|r| SpecExperiment { id: cell(r, 0), question: cell(r, 1), design: cell(r, 2), measurement: cell(r, 3), rule: cell(r, 4), kills: cell(r, 5), compute: cell(r, 6) })
        .collect();
    snap.gates = md_table(&md_section(&text, "### 14.1 "))
        .iter()
        .filter(|r| r.len() >= 4)
        .map(|r| SpecGate { id: cell(r, 0), rule: cell(r, 1), if_met: cell(r, 2), if_not: cell(r, 3) })
        .collect();
    snap.gpu_estimates = md_table(&md_section(&text, "### 14.3 "))
        .iter()
        .filter(|r| r.len() >= 4)
        .map(|r| GpuRow { label: cell(r, 0), counted: cell(r, 1), estimate: number(&cell(r, 2)), basis: cell(r, 3), ..Default::default() })
        .collect();
    snap.plan = md_table(&md_section(&text, "### 17.2 "))
        .iter()
        .filter(|r| r.len() >= 2)
        .map(|r| PlanRow { date: cell(r, 0), what: cell(r, 1) })
        .collect();
    for (name, empty) in [("experiment table (13)", snap.experiments.is_empty()), ("gate table (14.1)", snap.gates.is_empty()), ("dates table (17.2)", snap.plan.is_empty())] {
        if empty {
            snap.problems.push(format!("The specification's {name} could not be found"));
        }
    }
}

fn read_authorization(cfg: &Config, snap: &mut DocsSnap) {
    let path = cfg.study_dir.join("AUTHORIZATION.md");
    snap.authorization = info(cfg, &path);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            snap.allowances = md_table(&md_section(&text, "## 4. "))
                .iter()
                .filter(|r| r.len() >= 4)
                .map(|r| GpuRow { label: cell(r, 0), authorized: cell(r, 1), estimate: number(&cell(r, 2)), allowance: number(&cell(r, 3)), ..Default::default() })
                .collect();
        }
        Err(e) => snap.problems.push(format!("AUTHORIZATION.md could not be read: {e}.")),
    }
}

/// Rows of the addendum's compute table whose use is "GPU jobs": the account is the first
/// backticked word, the limit is the number before "GPUs".
pub fn parse_gpu_limits(text: &str) -> Vec<GpuLimit> {
    md_table(&md_section(text, "## 2. "))
        .iter()
        .filter(|r| r.len() >= 3 && r[0].trim() == "GPU jobs")
        .filter_map(|r| {
            let account = r[1].split('`').nth(1)?.to_string();
            let words: Vec<&str> = r[2].split_whitespace().collect();
            let at = words.iter().position(|w| w.trim_matches(',') == "GPUs")?;
            let gpus = words.get(at.checked_sub(1)?)?.parse().ok()?;
            Some(GpuLimit { account, gpus })
        })
        .collect()
}

fn read_addendum(cfg: &Config, snap: &mut DocsSnap) {
    let path = cfg.study_dir.join("AUTHORIZATION_ADDENDUM_1.md");
    snap.addendum = info(cfg, &path);
    if let Ok(text) = std::fs::read_to_string(&path) {
        snap.gpu_limits = parse_gpu_limits(&text);
    }
}

fn read_live_docs(cfg: &Config, snap: &mut DocsSnap) {
    let blocked = cfg.live_docs_dir.join("BLOCKED.md");
    snap.blocked_file = info(cfg, &blocked);
    if let Ok(text) = std::fs::read_to_string(&blocked) {
        snap.blocked = parse_blocked(&text);
    }

    let status = cfg.live_docs_dir.join("STATUS.md");
    snap.status_file = info(cfg, &status);
    if let Ok(text) = std::fs::read_to_string(&status) {
        snap.status_updated = text.lines().find(|l| l.starts_with("Updated")).map(|l| util::one_line(l, 120)).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let line = bullets(&lines).into_iter().find(|b| b.starts_with("Milestone:")).unwrap_or_default();
        snap.milestones = parse_milestones(line.trim_start_matches("Milestone:"));
        snap.status_milestone_line = util::one_line(&line, 200);
    }

    let ledger = cfg.live_docs_dir.join("RESEARCH_LEDGER.md");
    snap.ledger_file = info(cfg, &ledger);
    if let Ok(text) = std::fs::read_to_string(&ledger) {
        snap.register = md_table(&md_section(&text, "## 4. Experiment register"))
            .iter()
            .filter(|r| r.len() >= 2)
            .map(|r| RegisterRow { id: cell(r, 0), status: cell(r, 1), card: cell(r, 2), notes: cell(r, 3) })
            .collect();
    }

    let records = cfg.live_docs_dir.join("GATE_DECISIONS");
    snap.gate_records_dir = info(cfg, &records);
    let mut names: Vec<GateRecord> = std::fs::read_dir(&records)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| {
            let text = std::fs::read_to_string(e.path()).unwrap_or_default();
            parse_gate_record(&e.file_name().to_string_lossy(), &text, e.metadata().ok().and_then(|m| util::mtime_ms(&m)))
        })
        .collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    snap.gate_records = names;
}

fn read_cards(cfg: &Config, snap: &mut DocsSnap) {
    let dir = cfg.study_dir.join("EXPERIMENT_CARDS");
    let mut cards = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let lines: Vec<&str> = text.lines().collect();
        let title = lines.first().map(|l| l.trim_start_matches('#').trim().to_string()).unwrap_or_default();
        // The paragraph that says when the card was frozen, up to and including that sentence.
        let frozen = text
            .split("\n\n")
            .find(|p| p.contains("Frozen"))
            .map(|p| {
                let line = util::one_line(p, 400);
                let at = line.find("Frozen").unwrap_or(0);
                match line[at..].find(". ") {
                    Some(end) => line[..at + end + 1].to_string(),
                    None => line,
                }
            })
            .unwrap_or_default();
        let question = bullets(&lines)
            .into_iter()
            .find(|b| b.starts_with("Question:"))
            .map(|b| util::one_line(b.trim_start_matches("Question:"), 300))
            .unwrap_or_default();
        cards.push(Card {
            id: path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string(),
            title,
            frozen,
            question,
            path: cfg.short_path(&path.to_string_lossy()),
            mtime_ms: util::file_mtime_ms(&path),
        });
    }
    cards.sort_by(|a, b| a.id.cmp(&b.id));
    snap.cards = cards;
}

fn read_figure(cfg: &Config, snap: &mut DocsSnap) {
    snap.figure = ["regime_map.json", "gain_accounting.json", "regret_attribution.json"]
        .iter()
        .map(|name| {
            let path = cfg.figure_dir.join(name);
            let mut file = FigureFile { name: name.to_string(), path: cfg.short_path(&path.to_string_lossy()), ..Default::default() };
            match std::fs::metadata(&path) {
                Ok(meta) if meta.len() > 4 << 20 => {
                    file.exists = true;
                    file.error = "file is larger than 4 MB; not read".into();
                }
                Ok(meta) => {
                    file.exists = true;
                    file.mtime_ms = util::mtime_ms(&meta);
                    match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string())) {
                        Ok(v) => file.json = Some(v),
                        Err(e) => file.error = util::one_line(&e, 160),
                    }
                }
                Err(_) => {}
            }
            file
        })
        .collect();
}

pub fn read(cfg: &Config) -> DocsSnap {
    let mut snap = DocsSnap::default();
    read_spec(cfg, &mut snap);
    read_authorization(cfg, &mut snap);
    read_addendum(cfg, &mut snap);
    read_live_docs(cfg, &mut snap);
    read_cards(cfg, &mut snap);
    read_figure(cfg, &mut snap);
    snap
}

/// A cheap fingerprint of everything `read` depends on, to skip re-reading unchanged files.
pub fn fingerprint(cfg: &Config) -> String {
    let mut parts = Vec::new();
    let mut add = |p: std::path::PathBuf| {
        let meta = std::fs::metadata(&p).ok();
        parts.push(format!("{}:{:?}:{:?}", p.display(), meta.as_ref().map(|m| m.len()), meta.as_ref().and_then(util::mtime_ms)));
    };
    for name in ["SPEC_RESEARCH_ADVANCEMENT.md", "AUTHORIZATION.md", "AUTHORIZATION_ADDENDUM_1.md", "EXPERIMENT_CARDS"] {
        add(cfg.study_dir.join(name));
    }
    for name in ["BLOCKED.md", "STATUS.md", "RESEARCH_LEDGER.md", "GATE_DECISIONS"] {
        add(cfg.live_docs_dir.join(name));
    }
    for name in ["regime_map.json", "gain_accounting.json", "regret_attribution.json"] {
        add(cfg.figure_dir.join(name));
    }
    for entry in std::fs::read_dir(cfg.live_docs_dir.join("GATE_DECISIONS")).into_iter().flatten().flatten() {
        add(entry.path());
    }
    for entry in std::fs::read_dir(cfg.study_dir.join("EXPERIMENT_CARDS")).into_iter().flatten().flatten() {
        add(entry.path());
    }
    parts.join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_items_keep_their_status_and_request() {
        let text = "# BLOCKED\n\nintro\n\n## BLOCKED-001: sign the addendum (opened November 13, 2026)\n\n- What waits: GPU jobs on `cai5724`.\n  Nothing else waits.\n- Request, one step: paste this paragraph:\n\n  > I authorize it.\n\n- Meanwhile: every job runs on `cai6734`.\n- Status: CLOSED November 13, 2026, 2:55 AM EST. Confirmed.\n\n## BLOCKED-002: no cohort for E11 (opened November 13, 2026)\n\n- What waits: E11 only.\n- Status: OPEN.\n";
        let items = parse_blocked(text);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "BLOCKED-001");
        assert_eq!(items[0].title, "sign the addendum");
        assert_eq!(items[0].opened, "November 13, 2026");
        assert!(!items[0].open);
        assert!(items[0].waits.ends_with("Nothing else waits."));
        assert!(items[0].request.contains("I authorize it."));
        assert!(items[1].open);
        assert_eq!(items[1].waits, "E11 only.");
    }

    #[test]
    fn accepted_gaps_are_neither_open_nor_requests() {
        let text = "# BLOCKED\n\n## BLOCKED-003: something open (opened November 13, 2026)\n\n- Status: OPEN.\n\n# Accepted gaps\n\nLimits.\n\n## BLOCKED-002: no cohort for E11 (opened November 13, 2026)\n\n- What waits: E11 only.\n- Status: ACCEPTED GAP, closed November 13, 2026. Accepted by Daniel.\n";
        let items = parse_blocked(text);
        assert_eq!(items.len(), 2);
        assert!(items[0].open && !items[0].accepted);
        assert!(!items[1].open && items[1].accepted);
        assert!(items[1].status.starts_with("ACCEPTED GAP"));
        // An accepted status alone, outside the section, is enough.
        let alone = parse_blocked("## BLOCKED-009: x (opened today)\n\n- Status: ACCEPTED GAP, closed today.\n");
        assert!(!alone[0].open && alone[0].accepted);
    }

    #[test]
    fn gpu_limits_come_from_the_addendum_table() {
        let text = "# Addendum\n\n## 2. Compute added\n\n| Use | Account and QOS | Live limit |\n|---|---|---|\n| GPU jobs | `cai6734` on QOS `cai6734`, as before | 8 GPUs, 32 CPUs, 250 GB |\n| GPU jobs | `cai5724` on QOS `cai5724`, added | 8 GPUs, 32 CPUs, 250 GB |\n| CPU-only jobs | burst QOS `cai6734-b` | 288 CPUs, no GPUs |\n\n## 3. Next\n";
        assert_eq!(
            parse_gpu_limits(text),
            vec![GpuLimit { account: "cai6734".into(), gpus: 8 }, GpuLimit { account: "cai5724".into(), gpus: 8 }]
        );
    }

    #[test]
    fn gate_record_gives_result_and_path() {
        let text = "# Gate G0: M0 acceptance\n\nEvaluated November 13, 2026, about 4:00 AM EST, by the executing agent.\nImmutable once committed.\n\n## Rule\n\nSomething.\n\n## Result\n\nMET EXCEPT one item: the live part\nof launch check 7.4c.\n\n## Path taken\n\nM1 and M2 proceed.\n\nMore text.\n";
        let r = parse_gate_record("G0.md", text, Some(5));
        assert_eq!(r.title, "Gate G0: M0 acceptance");
        assert!(r.evaluated.starts_with("Evaluated November 13, 2026"));
        assert_eq!(r.result, "MET EXCEPT one item: the live part of launch check 7.4c.");
        assert_eq!(r.path, "M1 and M2 proceed.");
    }

    #[test]
    fn milestone_line_expands_ranges() {
        let got = parse_milestones(" M0 IN PROGRESS. M1 to M5 NOT STARTED.");
        let flat: Vec<String> = got.iter().map(|m| format!("{}={}", m.id, m.state)).collect();
        assert_eq!(flat, ["M0=IN PROGRESS", "M1=NOT STARTED", "M2=NOT STARTED", "M3=NOT STARTED", "M4=NOT STARTED", "M5=NOT STARTED"]);
        let got = parse_milestones("M0 and M1 COMPLETE; M2 IN PROGRESS; M3 to M5 NOT STARTED");
        assert_eq!(got.len(), 6);
        assert_eq!(got[1].state, "COMPLETE");
        assert_eq!(got[2].state, "IN PROGRESS");
    }
}
