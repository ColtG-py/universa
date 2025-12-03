#!/bin/bash
# Run the Universa API server

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default port
PORT=${API_PORT:-8000}
HOST=${API_HOST:-0.0.0.0}

# Run uvicorn
echo "Starting Universa API on ${HOST}:${PORT}..."
uvicorn api.main:app --host $HOST --port $PORT --reload
