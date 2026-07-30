#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
CACHE_ROOT="${SERVER_CACHE_ROOT:-$REPO_ROOT/.cache}"
HF_HOME="$CACHE_ROOT/huggingface"
PIP_CACHE_DIR="$CACHE_ROOT/pip"
TORCH_HOME="$CACHE_ROOT/torch"
TRITON_CACHE_DIR="$CACHE_ROOT/triton"
CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
XDG_CACHE_HOME="$CACHE_ROOT/xdg"
TMPDIR="$CACHE_ROOT/tmp"
HF_DATASETS_CACHE="$HF_HOME/datasets"
NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch_inductor"
WANDB_DIR="$CACHE_ROOT/wandb"
WANDB_CACHE_DIR="$CACHE_ROOT/wandb_cache"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TMPDIR" "$HF_DATASETS_CACHE" \
  "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR"
export CACHE_ROOT HF_HOME PIP_CACHE_DIR TORCH_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH XDG_CACHE_HOME TMPDIR
export HF_DATASETS_CACHE NUMBA_CACHE_DIR TORCH_EXTENSIONS_DIR TORCHINDUCTOR_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR

resolved_venv="$(realpath -m "$VENV_DIR")"
resolved_cache="$(realpath -m "$CACHE_ROOT")"
for storage_path in "$resolved_venv" "$resolved_cache"; do
  case "$storage_path" in
    "$REPO_ROOT"|"$REPO_ROOT"/*) ;;
    *)
      echo "Refusing to write experiment storage outside $REPO_ROOT: $storage_path" >&2
      exit 2
      ;;
  esac
done

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
