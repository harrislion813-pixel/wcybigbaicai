@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call "安装桌面版依赖.bat"
endlocal
