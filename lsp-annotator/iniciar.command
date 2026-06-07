#!/bin/bash
cd "$(dirname "$0")"
echo "Iniciando LSP Annotator..."
uv run python main.py &
sleep 2
open http://localhost:8000
wait
