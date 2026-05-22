#!/usr/bin/env bash
set -euo pipefail

mkdir -p /output /logs
cd /app

/app/.venv/bin/python /app/main.py 2>&1 | tee /logs/runtime.log
