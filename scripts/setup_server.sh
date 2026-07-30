#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.10|3.11|3.12) ;;
  *)
    echo "Python $PYTHON_VERSION is unsupported. Use Python 3.10, 3.11, or 3.12 via PYTHON_BIN." >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements-train.txt
python -m pip install -e .
python scripts/preflight_server.py --data_only

echo "Environment ready. Activate it with: source $VENV_DIR/bin/activate"
