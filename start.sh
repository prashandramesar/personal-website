#!/bin/bash
# start.sh for single process solution

# Set up error handling
set -e

echo "Starting deployment process..."

# Make sure the uploads directory exists
mkdir -p static/uploads
echo "Created uploads directory"

# Set environment variables
export PRODUCTION=true

# Start the application with uvicorn
echo "Starting combined Flask/FastAPI application..."
exec uvicorn app:app --host=0.0.0.0 --port=${PORT:-10000} --log-level=info
