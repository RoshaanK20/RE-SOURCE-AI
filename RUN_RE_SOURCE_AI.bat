@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating RE-SOURCE AI application environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -m pip show streamlit >nul 2>&1
if errorlevel 1 (
  echo Installing the application's declared dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)
echo Starting RE-SOURCE AI at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0
:error
echo.
echo Setup or startup failed. Read START_HERE.txt or inspect this terminal output.
pause
exit /b 1
