#!/bin/bash

# Exit on error
set -e

echo "=========================================="
echo "🚀 Starting Hybrid RAG + Ollama System"
echo "=========================================="

# 1. Check if virtualenv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install requirements
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Check Ollama
if command -v ollama &> /dev/null; then
    echo "🦙 Ollama is installed. Checking if service is running..."
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "⚠️ Ollama daemon not running. Starting ollama serve in background..."
        ollama serve &
        sleep 3
    else
        echo "✅ Ollama service is active."
    fi
else
    echo "⚠️ Ollama is not installed in standard PATH. Please make sure Ollama is installed and running."
fi

# 5. Start FastAPI Backend in background
echo "⚡ Starting FastAPI Backend on http://localhost:8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Trap cleanup to kill background processes on exit
trap "kill $BACKEND_PID" EXIT

sleep 2

# 6. Start Streamlit Frontend
echo "✨ Starting Streamlit UI on http://localhost:8501..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
