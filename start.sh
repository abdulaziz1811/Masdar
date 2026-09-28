#!/usr/bin/env bash
# Start Masdar's chat page: sets up a local environment the first time,
# then opens http://127.0.0.1:8000 in the browser.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.11 أو أحدث غير مثبّت. ثبّته من https://www.python.org/downloads/ ثم أعد التشغيل."
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "مطلوب Python 3.11 أو أحدث (الموجود: $("$PY" --version 2>&1))."
  exit 1
fi

if [ ! -f .venv/.masdar-ready ]; then
  echo "تجهيز البيئة لأول مرة (دقيقة أو دقيقتان)…"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -e ".[ai]"
  touch .venv/.masdar-ready
fi

exec .venv/bin/python -m masdar.cli serve --open "$@"
