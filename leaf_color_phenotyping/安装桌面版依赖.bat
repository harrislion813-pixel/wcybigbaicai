@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo 未检测到 Python。请先安装 Python 3.10 或更高版本，并勾选 Add Python to PATH。
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 正在创建项目虚拟环境...
  python -m venv .venv
  if errorlevel 1 goto :failed
)
echo 正在安装桌面版依赖，请保持网络连接...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-gui.txt
if errorlevel 1 goto :failed
echo.
echo 安装完成。现在可以双击“启动桌面版.bat”。
pause
exit /b 0

:failed
echo.
echo 安装失败，请检查上方错误信息。
pause
exit /b 1
