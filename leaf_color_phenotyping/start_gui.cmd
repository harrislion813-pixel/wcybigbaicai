@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo 未找到桌面版运行环境，请先运行 install_gui.cmd。
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
endlocal
