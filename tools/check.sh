#!/usr/bin/env bash
# Local pre-commit check: byte-compile + full smoke test (uses the venv).
cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
echo "=== py_compile ==="
$PY -m py_compile app.py tools/smoke_test.py tools/check_klein_extras.py || { echo "COMPILE FAILED"; exit 1; }
echo "=== config JSON ==="
$PY -c "import json; json.load(open('config-sample.txt',encoding='utf-8')); print('config-sample.txt OK')" || exit 1
echo "=== smoke test ==="
$PY tools/smoke_test.py
