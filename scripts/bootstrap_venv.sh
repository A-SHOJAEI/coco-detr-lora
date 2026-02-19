#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found" >&2
  exit 1
fi

mkdir -p .venv

# Create a venv without pip, then bootstrap pip using get-pip.py.
"$PYTHON_BIN" -m venv --without-pip .venv

GETPIP_URL="https://bootstrap.pypa.io/get-pip.py"
GETPIP_PATH=".venv/get-pip.py"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$GETPIP_URL" -o "$GETPIP_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$GETPIP_PATH" "$GETPIP_URL"
else
  echo "error: need curl or wget to download get-pip.py" >&2
  exit 1
fi

.venv/bin/python "$GETPIP_PATH"

# Keep pip tooling stable across machines.
.venv/bin/pip install --disable-pip-version-check --no-input "pip==24.0" "setuptools==69.5.1" "wheel==0.43.0"
