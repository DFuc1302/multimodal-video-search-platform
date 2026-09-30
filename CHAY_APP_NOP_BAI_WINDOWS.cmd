@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_submission_server_windows.ps1"
if errorlevel 1 (
  echo.
  echo Khong khoi dong duoc app. Xem thong bao loi ben tren.
  pause
)
