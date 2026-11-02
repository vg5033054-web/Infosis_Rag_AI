#!/usr/bin/env bash
# Infosys Enterprise RAG Assistant Launcher Script

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "========================================================"
echo " Starting Infosys Enterprise RAG AI Assistant..."
echo " Grounded in 752 pages across FY24-25 & FY25-26 Reports"
echo " Powered by Groq LPU Inference"
echo "========================================================"
echo " Web UI running at: http://localhost:8000"
echo " Press Ctrl+C to stop the server."
echo "========================================================"

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
