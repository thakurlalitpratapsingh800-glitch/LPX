@echo off
REM ============================================================
REM  Askwise AI - one-click Windows setup (all free software)
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo [1/6] Creating Python virtual environment...
if not exist .venv (
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat

echo.
echo [2/6] Installing free Python packages (Flask, ddgs, pypdf, torch, psutil)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || goto :error

echo.
echo [3/6] Checking Ollama (free local AI server)...
where ollama >nul 2>nul
if errorlevel 1 (
  echo   Ollama is not installed yet.
  echo   Install it FREE from: https://ollama.com/download/windows
  echo   Or run:  winget install --id Ollama.Ollama --exact --source winget
  echo   After installing, open a NEW terminal and re-run this script.
) else (
  echo   Ollama found. Pulling the best free model for your RAM...
  for /f "delims=" %%A in ('python -c "import psutil;print(round(psutil.virtual_memory().total/2**30))"') do set RAMGB=%%A
  set MODEL=llama3.2:1b
  if !RAMGB! GEQ 14 set MODEL=llama3.2:3b
  if !RAMGB! GEQ 30 set MODEL=llama3.1:8b
  echo   !RAMGB! GB RAM detected -^> pulling !MODEL! ...
  ollama pull !MODEL!
)

echo.
echo [4/6] Preparing your own model's dataset (data folder)...
python -m askwise_model.dataset --data data || goto :error

echo.
echo [5/6] Smoke-test training your own Transformer (200 steps, CPU/GPU auto)...
python -m askwise_model.train --data data --steps 200 --log-interval 50 --eval-interval 100 || goto :error

echo.
echo [6/6] Generating a sample text with YOUR trained model...
python -m askwise_model.generate --prompt "artificial intelligence" --tokens 40

echo.
echo ============================================================
echo  Setup complete! Next steps:
echo    1. Start the server:   python app.py
echo    2. Open               http://127.0.0.1:5000
echo    3. For full training:  python -m askwise_model.train --data data --steps 5000
echo    4. For offline chat:   make sure Ollama is running (ollama serve)
echo ============================================================
goto :eof

:error
echo.
echo  ERROR: setup failed on the step above. Read the message and try again.
exit /b 1
