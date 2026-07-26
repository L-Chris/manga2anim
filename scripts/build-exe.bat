@echo off
REM Build the Tauri desktop bundle (debug profile = fast, no LTO; same app).
REM This .bat sidesteps Git Bash's POSIX-vs-Windows PATH mismatch: bash can run
REM node fine, but the native node -> tauri.exe -> cargo spawn chain needs a
REM Windows-style PATH, which bash's /c quoting mangles. A .bat runs natively.
set PATH=C:\Users\Chris\.cargo\bin;C:\nvm4w\nodejs;C:\Windows\system32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0
cd /d F:\projects\manga2anim
C:\nvm4w\nodejs\node.exe node_modules\@tauri-apps\cli\tauri.js build --debug
exit /b %ERRORLEVEL%
