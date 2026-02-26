#!/bin/bash
# Start the local LLM server in the background
python scripts/LLM_server.py --host 0.0.0.0 --port 11434 &

# Start the main API server in the foreground
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
