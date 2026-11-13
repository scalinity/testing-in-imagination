//! TII monitor: a read-only window onto study TII-CEE-1.
//!
//! The webview gets three commands and nothing else: no shell, no filesystem, no network.
//! All file, git and ssh access happens in this process.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod docs;
mod engine;
mod gitinfo;
mod link;
mod remote;
mod transcripts;
mod util;

use engine::{Engine, Known, StateReply};
use std::sync::Arc;
use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::{Manager, State};

/// Everything on screen. `known` holds the revisions the webview already has, so unchanged
/// parts are not sent again; `window` is the worker timeline's width in seconds.
#[tauri::command(async)]
fn get_state(engine: State<'_, Arc<Engine>>, known: Known, window: u64) -> StateReply {
    engine.state(known, window, false)
}

/// Sets how often HiPerGator is read. Clamped to 60 seconds at the least; returns the value in force.
#[tauri::command]
fn set_remote_interval(engine: State<'_, Arc<Engine>>, seconds: u64) -> u64 {
    engine.set_interval(seconds)
}

/// Asks for a read of HiPerGator as soon as the 60-second floor allows.
#[tauri::command]
fn read_cluster_now(engine: State<'_, Arc<Engine>>) {
    engine.request_read();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let demo = args.iter().any(|a| a == "--demo") || std::env::var("TII_MONITOR_DEMO").is_ok_and(|v| v == "1");
    let engine = Arc::new(Engine::new(config::Config::load(demo)));

    // Development aid: `--dump <dir>` writes one snapshot as JSON and exits.
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        let dir = args.get(i + 1).map(std::path::PathBuf::from).unwrap_or_else(|| engine.cfg.repo.join("monitor/.snapshot"));
        match engine.dump(&dir, 3600) {
            Ok(()) => println!("wrote {}", dir.join("state.json").display()),
            Err(e) => {
                eprintln!("could not write the snapshot: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    engine.clone().start();
    tauri::Builder::default()
        .manage(engine)
        .setup(|app| {
            // The window has no title bar, so Close and Minimize are wired to the window directly.
            let close = MenuItemBuilder::with_id("close", "Close Window").accelerator("CmdOrCtrl+W").build(app)?;
            let minimize = MenuItemBuilder::with_id("minimize", "Minimize").accelerator("CmdOrCtrl+M").build(app)?;
            let app_menu = SubmenuBuilder::new(app, "TII Monitor").hide().hide_others().show_all().separator().quit().build()?;
            let edit_menu = SubmenuBuilder::new(app, "Edit").copy().select_all().build()?;
            let window_menu = SubmenuBuilder::new(app, "Window").item(&minimize).fullscreen().separator().item(&close).build()?;
            let menu = MenuBuilder::new(app).items(&[&app_menu, &edit_menu, &window_menu]).build()?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let Some(window) = app.get_webview_window("main") else { return };
                match event.id().as_ref() {
                    "close" => {
                        let _ = window.close();
                    }
                    "minimize" => {
                        let _ = window.minimize();
                    }
                    _ => {}
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_state, set_remote_interval, read_cluster_now])
        .run(tauri::generate_context!())
        .expect("the monitor window could not start");
}
