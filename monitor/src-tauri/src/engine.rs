//! Collects every source on background threads and hands the webview one snapshot.
//!
//! Cadences (also in DESIGN.md):
//!   transcripts        every 1 s  (stat, then read only the appended bytes)
//!   git, study docs    every 2 s  (three read-only git commands; stat fingerprint for docs)
//!   keeper log         every 5 s
//!   SSH master check   every 20 s (`ssh -O check`, local) and before every cluster read
//!   launchd jobs       every 20 s
//!   HiPerGator files   every 120 s by default, never closer than 60 s
//! A part's revision changes only when its content does, so the webview is not re-sent, and
//! does not re-render, data that stayed the same.

use crate::config::{Config, DEFAULT_REMOTE_INTERVAL_S, MIN_REMOTE_INTERVAL_S};
use crate::docs::DocsSnap;
use crate::gitinfo::GitSnap;
use crate::link::LinkSnap;
use crate::remote::{Remote, RemoteData, HOST, ROOT};
use crate::transcripts::{Store, WorkersView};
use crate::{docs, gitinfo, link, util};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_REMOTE_INTERVAL_S: u64 = 3600;
/// Fallback for the study's start: the author time of commit 632bd18.
const STUDY_START_FALLBACK_MS: i64 = 1_794_555_460_000;

#[derive(Default)]
struct Versioned<T> {
    rev: u64,
    value: T,
    json: String,
}

impl<T: Serialize> Versioned<T> {
    /// Stores a new value; the revision moves only if the content differs.
    fn put(&mut self, value: T) {
        let json = serde_json::to_string(&value).unwrap_or_default();
        if self.rev == 0 || json != self.json {
            self.rev += 1;
            self.json = json;
            self.value = value;
        }
    }
}

/// When each source was last looked at, on the monitor's clock.
#[derive(Serialize, Default, Clone)]
pub struct Checked {
    pub transcripts: Option<i64>,
    pub git: Option<i64>,
    pub docs: Option<i64>,
    pub keeper: Option<i64>,
    pub master: Option<i64>,
    pub launchd: Option<i64>,
}

pub struct Engine {
    pub cfg: Config,
    /// Author time of the study's first commit; read from git off the main thread.
    study_start_ms: AtomicI64,
    transcripts: Mutex<Store>,
    link: Mutex<Versioned<LinkSnap>>,
    git: Mutex<Versioned<GitSnap>>,
    docs: Mutex<Versioned<DocsSnap>>,
    remote: Mutex<Versioned<RemoteData>>,
    checked: Mutex<Checked>,
    remote_reading: AtomicBool,
    remote_last_start_real_ms: AtomicI64,
    interval_s: AtomicU64,
    read_now: AtomicBool,
}

#[derive(Deserialize, Default)]
pub struct Known {
    #[serde(default)]
    pub link: u64,
    #[serde(default)]
    pub git: u64,
    #[serde(default)]
    pub docs: u64,
    #[serde(default)]
    pub remote: u64,
}

#[derive(Serialize)]
pub struct Revs {
    pub link: u64,
    pub git: u64,
    pub docs: u64,
    pub remote: u64,
}

#[derive(Serialize)]
pub struct RemoteMeta {
    pub host: &'static str,
    pub root: &'static str,
    pub reading: bool,
    pub interval_s: u64,
    pub min_interval_s: u64,
    pub max_interval_s: u64,
    /// Seconds until the next scheduled read; measured on the real clock.
    pub next_read_in_s: Option<u64>,
    /// Seconds until a manual read is allowed (the 60-second floor).
    pub manual_wait_s: u64,
}

#[derive(Serialize)]
pub struct StateReply {
    pub now_ms: i64,
    pub demo: bool,
    pub repo: String,
    pub study_start_ms: i64,
    pub revs: Revs,
    pub checked: Checked,
    pub workers: WorkersView,
    pub link: Option<LinkSnap>,
    pub git: Option<GitSnap>,
    pub docs: Option<DocsSnap>,
    pub remote: Option<RemoteData>,
    pub remote_meta: RemoteMeta,
}

#[derive(Serialize, Deserialize, Default)]
struct Settings {
    remote_interval_s: Option<u64>,
}

fn clamp_interval(s: u64) -> u64 {
    s.clamp(MIN_REMOTE_INTERVAL_S, MAX_REMOTE_INTERVAL_S)
}

impl Engine {
    pub fn new(cfg: Config) -> Engine {
        let settings: Settings = std::fs::read_to_string(cfg.data_dir.join("settings.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        Engine {
            study_start_ms: AtomicI64::new(if cfg.demo { 0 } else { STUDY_START_FALLBACK_MS }),
            transcripts: Mutex::new(Store::new()),
            link: Mutex::default(),
            git: Mutex::default(),
            docs: Mutex::default(),
            remote: Mutex::default(),
            checked: Mutex::default(),
            remote_reading: AtomicBool::new(false),
            remote_last_start_real_ms: AtomicI64::new(0),
            interval_s: AtomicU64::new(clamp_interval(settings.remote_interval_s.unwrap_or(DEFAULT_REMOTE_INTERVAL_S))),
            read_now: AtomicBool::new(false),
            cfg,
        }
    }

    pub fn set_interval(&self, seconds: u64) -> u64 {
        let s = clamp_interval(seconds);
        self.interval_s.store(s, Ordering::Relaxed);
        if !self.cfg.demo {
            let _ = std::fs::create_dir_all(&self.cfg.data_dir);
            let _ = std::fs::write(
                self.cfg.data_dir.join("settings.json"),
                serde_json::to_string(&Settings { remote_interval_s: Some(s) }).unwrap_or_default(),
            );
        }
        s
    }

    pub fn request_read(&self) {
        self.read_now.store(true, Ordering::Relaxed);
    }

    fn read_study_start(&self) {
        if !self.cfg.demo {
            if let Some(ms) = gitinfo::study_start_ms(&self.cfg) {
                self.study_start_ms.store(ms, Ordering::Relaxed);
            }
        }
    }

    fn collect_transcripts(&self) {
        self.transcripts.lock().unwrap().scan(&self.cfg, self.study_start_ms.load(Ordering::Relaxed));
        self.checked.lock().unwrap().transcripts = Some(self.cfg.now_ms());
    }

    fn collect_keeper(&self) {
        let mut snap = self.link.lock().unwrap().value.clone();
        link::read_keeper(&self.cfg, &mut snap);
        self.merge_link(|slot| {
            slot.keeper = snap.keeper;
            slot.keeper_log = snap.keeper_log;
            slot.keeper_log_exists = snap.keeper_log_exists;
            slot.keeper_log_mtime_ms = snap.keeper_log_mtime_ms;
            slot.keeper_unparsed = snap.keeper_unparsed;
        });
        self.checked.lock().unwrap().keeper = Some(self.cfg.now_ms());
    }

    fn collect_master(&self) -> (bool, String) {
        let mut snap = LinkSnap::default();
        let result = link::read_master(&self.cfg, &mut snap);
        self.merge_link(|slot| {
            slot.master_up = snap.master_up;
            slot.master_detail = snap.master_detail;
        });
        self.checked.lock().unwrap().master = Some(self.cfg.now_ms());
        result
    }

    fn collect_launchd(&self) {
        let mut snap = LinkSnap::default();
        link::read_launchd(&self.cfg, &mut snap);
        self.merge_link(|slot| slot.launchd = snap.launchd);
        self.checked.lock().unwrap().launchd = Some(self.cfg.now_ms());
    }

    fn merge_link(&self, change: impl FnOnce(&mut LinkSnap)) {
        let mut slot = self.link.lock().unwrap();
        let mut next = slot.value.clone();
        change(&mut next);
        slot.put(next);
    }

    fn collect_git(&self) {
        let snap = gitinfo::read(&self.cfg);
        self.git.lock().unwrap().put(snap);
        self.checked.lock().unwrap().git = Some(self.cfg.now_ms());
    }

    fn collect_docs(&self, last_fingerprint: &mut String) {
        let fingerprint = docs::fingerprint(&self.cfg);
        if &fingerprint != last_fingerprint || self.docs.lock().unwrap().rev == 0 {
            let snap = docs::read(&self.cfg);
            *last_fingerprint = fingerprint;
            self.docs.lock().unwrap().put(snap);
        }
        self.checked.lock().unwrap().docs = Some(self.cfg.now_ms());
    }

    /// One read of the cluster, if the rules allow one now. Returns false when nothing was sent.
    fn remote_cycle(&self, remote: &mut Remote) -> bool {
        self.remote_reading.store(true, Ordering::Relaxed);
        let master = self.collect_master();
        let ran = remote.cycle(self.cfg.now_ms(), util::real_now_ms(), master);
        self.publish_remote(remote);
        self.remote_reading.store(false, Ordering::Relaxed);
        ran
    }

    fn publish_remote(&self, remote: &Remote) {
        self.remote_last_start_real_ms.store(remote.last_start_ms.unwrap_or(0), Ordering::Relaxed);
        self.remote.lock().unwrap().put(remote.data.clone());
    }

    /// Starts the collectors. Each family has its own thread, so that a source which blocks
    /// (the repository, while macOS waits for a permission answer) cannot stall the others.
    pub fn start(self: Arc<Self>) {
        let fast = self.clone();
        std::thread::spawn(move || loop {
            fast.collect_transcripts();
            std::thread::sleep(Duration::from_secs(1));
        });
        let repo = self.clone();
        std::thread::spawn(move || {
            repo.read_study_start();
            let mut fingerprint = String::new();
            loop {
                repo.collect_git();
                repo.collect_docs(&mut fingerprint);
                std::thread::sleep(Duration::from_secs(2));
            }
        });
        let local = self.clone();
        std::thread::spawn(move || {
            let mut tick: u64 = 0;
            loop {
                local.collect_keeper();
                if tick % 4 == 0 {
                    local.collect_master();
                    local.collect_launchd();
                }
                tick += 1;
                std::thread::sleep(Duration::from_secs(5));
            }
        });
        let remote_side = self.clone();
        std::thread::spawn(move || {
            let mut remote = Remote::new(&remote_side.cfg);
            remote_side.publish_remote(&remote);
            // When the master is down nothing is sent; look again (locally) in 15 seconds.
            let mut not_before_real_ms = 0i64;
            loop {
                let real_now = util::real_now_ms();
                let interval = remote_side.interval_s.load(Ordering::Relaxed);
                let due = remote.wait_s(real_now, interval) == 0;
                let asked = remote_side.read_now.load(Ordering::Relaxed) && remote.wait_s(real_now, MIN_REMOTE_INTERVAL_S) == 0;
                if (due || asked) && real_now >= not_before_real_ms {
                    remote_side.read_now.store(false, Ordering::Relaxed);
                    if !remote_side.remote_cycle(&mut remote) {
                        not_before_real_ms = real_now + 15_000;
                    }
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        });
    }

    pub fn state(&self, known: Known, window_s: u64, everything: bool) -> StateReply {
        let take = |rev: u64, have: u64| everything || rev != have;
        let link = self.link.lock().unwrap();
        let git = self.git.lock().unwrap();
        let docs = self.docs.lock().unwrap();
        let remote = self.remote.lock().unwrap();
        let interval = self.interval_s.load(Ordering::Relaxed);
        let last_start = self.remote_last_start_real_ms.load(Ordering::Relaxed);
        let since_s = (last_start > 0).then(|| ((util::real_now_ms() - last_start).max(0) / 1000) as u64);
        StateReply {
            now_ms: self.cfg.now_ms(),
            demo: self.cfg.demo,
            repo: self.cfg.repo.to_string_lossy().into_owned(),
            study_start_ms: self.study_start_ms.load(Ordering::Relaxed),
            revs: Revs { link: link.rev, git: git.rev, docs: docs.rev, remote: remote.rev },
            checked: self.checked.lock().unwrap().clone(),
            workers: self.transcripts.lock().unwrap().view(&self.cfg, window_s),
            link: take(link.rev, known.link).then(|| link.value.clone()),
            git: take(git.rev, known.git).then(|| git.value.clone()),
            docs: take(docs.rev, known.docs).then(|| docs.value.clone()),
            remote: take(remote.rev, known.remote).then(|| remote.value.clone()),
            remote_meta: RemoteMeta {
                host: HOST,
                root: ROOT,
                reading: self.remote_reading.load(Ordering::Relaxed),
                interval_s: interval,
                min_interval_s: MIN_REMOTE_INTERVAL_S,
                max_interval_s: MAX_REMOTE_INTERVAL_S,
                next_read_in_s: since_s.map(|s| interval.saturating_sub(s)),
                manual_wait_s: since_s.map_or(0, |s| MIN_REMOTE_INTERVAL_S.saturating_sub(s)),
            },
        }
    }

    /// Development aid: collect every source once and write one snapshot file.
    pub fn dump(self: Arc<Self>, dir: &std::path::Path, window_s: u64) -> std::io::Result<()> {
        self.read_study_start();
        self.collect_transcripts();
        self.collect_keeper();
        self.collect_launchd();
        self.collect_git();
        self.collect_docs(&mut String::new());
        let mut remote = Remote::new(&self.cfg);
        let _ = self.remote_cycle(&mut remote);
        std::fs::create_dir_all(dir)?;
        let reply = self.state(Known::default(), window_s, true);
        std::fs::write(dir.join("state.json"), serde_json::to_vec(&reply)?)
    }
}
