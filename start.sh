#!/bin/bash
# start.sh - Startup script for Render.com

# Set up error handling
set -e

echo "Starting deployment process..."

# Check if we're running on Render
if [ -n "$RENDER" ]; then
  echo "Running on Render.com"
  export PRODUCTION=true
fi

# Make sure the uploads directory exists
mkdir -p static/uploads
echo "Created uploads directory"

# Start FastAPI in the background with proper logging
echo "Starting FastAPI service on port $FASTAPI_PORT..."
python -c "from app import dog_api; import uvicorn; uvicorn.run(dog_api, host='0.0.0.0', port=$FASTAPI_PORT)" > fastapi.log 2>&1 &
FASTAPI_PID=$!

# Give FastAPI time to start up
echo "Waiting for FastAPI to initialize..."
sleep 5

# Check if FastAPI started successfully
if ps -p $FASTAPI_PID > /dev/null; then
  echo "FastAPI started successfully (PID: $FASTAPI_PID)"
else
  echo "WARNING: FastAPI failed to start. Check logs for details."
  cat fastapi.log
fi

# Start Flask app (gunicorn)
echo "Starting Flask app with gunicorn..."
exec gunicorn app:app --log-file=- --access-logfile=- --workers=2 --threads=4 --timeout=120
