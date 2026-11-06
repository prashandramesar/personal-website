#!/bin/bash
# Starts the app on Render.
set -e

exec uvicorn app:app --host=0.0.0.0 --port="${PORT:-10000}" --log-level=info
