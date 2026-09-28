@echo off
rem Start Masdar's chat page: sets up a local environment the first time,
rem then opens http://127.0.0.1:8000 in the browser.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=py -3"
%PY% --version >nul 2>&1 || set "PY=python"
%PY% -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 or newer is required: https://www.python.org/downloads/
  echo During setup, tick "Add python.exe to PATH".
  pause
  exit /b 1
)

if not exist ".venv\.masdar-ready" (
  echo First run: preparing the environment, this takes a minute or two...
  %PY% -m venv .venv || goto :failed
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || goto :failed
  ".venv\Scripts\python.exe" -m pip install --quiet -e ".[ai]" || goto :failed
  type nul > ".venv\.masdar-ready"
)

set PYTHONUTF8=1
".venv\Scripts\python.exe" -m masdar.cli serve --open %*
pause
exit /b 0

:failed
echo Setup failed. Check the internet connection and try again.
pause
exit /b 1
