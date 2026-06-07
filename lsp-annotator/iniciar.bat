@echo off
echo Iniciando LSP Annotator...
cd /d "%~dp0"
start /B uv run python main.py
timeout /t 2 /nobreak >nul
start http://localhost:8000
