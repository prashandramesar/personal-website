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

# Set the FastAPI port
export FASTAPI_PORT=${FASTAPI_PORT:-8000}

# Note: We are NOT starting FastAPI as a separate process here
# because we're now starting it within the app.py script for better process management

# Start Flask app (gunicorn)
echo "Starting Flask app with gunicorn..."
exec gunicorn app:app --log-file=- --access-logfile=- --workers=2 --threads=4 --timeout=120
