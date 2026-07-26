// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

/// Returns the absolute path to the directory where ONNX model weights are
/// expected to live. The frontend uses this to locate yolo26s / pp-ocrv6
/// weights that the user drops in. Falls back to the app config dir.
#[tauri::command]
fn models_dir(app: tauri::AppHandle) -> Result<String, String> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    // Best-effort create so the user has a folder to drop weights into.
    let _ = std::fs::create_dir_all(&dir);
    Ok(dir.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![models_dir])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
