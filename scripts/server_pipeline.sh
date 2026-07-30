#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_FILE="${SERVER_CONFIG_FILE:-.server.env}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

load_config() {
  if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    source "$CONFIG_FILE"
    set +a
  fi
  BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-0.5B}"
  BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-main}"
  SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/rebuttal_v2/source/train_clean.jsonl}"
  SFT_OUTPUT="${SFT_OUTPUT:-outputs/rebuttal_sft/qwen2_0.5b_op10}"
  EXPERIMENT_SUITE="${EXPERIMENT_SUITE:-full}"
  SEEDS="${SEEDS:-414,6201,2026}"
  RUN_ROOT="${RUN_ROOT:-outputs/rebuttal_taxonomy_full}"
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
}

require_workspace_storage() {
  local name path resolved
  while (( $# )); do
    name="$1"
    path="$2"
    shift 2
    resolved="$(realpath -m "$path")"
    case "$resolved" in
      "$REPO_ROOT"|"$REPO_ROOT"/*) ;;
      *)
        echo "$name must stay under the data-disk workspace $REPO_ROOT; got $resolved" >&2
        exit 2
        ;;
    esac
  done
}

activate_environment() {
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing $VENV_DIR. Run bootstrap or bootstrap-sft first." >&2
    exit 2
  fi
  source "$VENV_DIR/bin/activate"
}

require_checkpoint() {
  if [[ -z "${SFT_CHECKPOINT:-}" || ! -d "$SFT_CHECKPOINT" ]]; then
    echo "SFT_CHECKPOINT is missing or invalid. Re-run bootstrap with the checkpoint path." >&2
    exit 2
  fi
}

save_config() {
  local checkpoint="$1"
  local model="$2"
  local revision="${3:-main}"
  {
    printf 'SFT_CHECKPOINT=%q\n' "$checkpoint"
    printf 'BASE_MODEL=%q\n' "$model"
    printf 'BASE_MODEL_REVISION=%q\n' "$revision"
    printf 'SFT_TRAIN_FILE=%q\n' 'data/rebuttal_v2/source/train_clean.jsonl'
    printf 'SFT_OUTPUT=%q\n' "$checkpoint"
    printf 'SERVER_CACHE_ROOT=%q\n' "$REPO_ROOT/.cache"
    printf 'EXPERIMENT_SUITE=full\n'
    printf 'SEEDS=414,6201,2026\n'
    printf 'RUN_ROOT=outputs/rebuttal_taxonomy_full\n'
  } > "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server_pipeline.sh inspect
  bash scripts/server_pipeline.sh bootstrap /path/to/sft_checkpoint [base_model]
  bash scripts/server_pipeline.sh bootstrap-sft [base_model] [sft_output]
  bash scripts/server_pipeline.sh start [base_model] [sft_output]
  bash scripts/server_pipeline.sh smoke
  bash scripts/server_pipeline.sh launch-full
  bash scripts/server_pipeline.sh status
  bash scripts/server_pipeline.sh test

The start command prepares everything, trains SFT, runs smoke, and launches full in the background.
The full command trains/resumes the matrix and evaluates validation only.
The held-out test split is run separately with the test command.
EOF
}

mode="${1:-help}"
load_config
require_workspace_storage \
  VENV_DIR "$VENV_DIR" \
  CONFIG_FILE "$CONFIG_FILE" \
  CACHE_ROOT "$CACHE_ROOT" \
  TMPDIR "$TMPDIR" \
  RUN_ROOT "$RUN_ROOT"

case "$mode" in
  inspect)
    mkdir -p outputs/server_preflight
    python3 scripts/inspect_server.py --out outputs/server_preflight/environment.json
    ;;
  bootstrap)
    checkpoint="${2:-${SFT_CHECKPOINT:-}}"
    model="${3:-$BASE_MODEL}"
    if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
      echo "Provide an existing SFT checkpoint directory as argument 2." >&2
      exit 2
    fi
    save_config "$checkpoint" "$model" "${BASE_MODEL_REVISION:-main}"
    load_config
    mkdir -p outputs/server_preflight
    python3 scripts/inspect_server.py --out outputs/server_preflight/environment_before_setup.json
    bash scripts/setup_server.sh
    activate_environment
    python scripts/preflight_server.py --checkpoint "$SFT_CHECKPOINT" --out outputs/server_preflight/preflight.json
    python -m unittest discover -s rebuttal/tests -v
    echo "Bootstrap complete. Next: bash scripts/server_pipeline.sh smoke"
    ;;
  bootstrap-sft)
    model="${2:-$BASE_MODEL}"
    checkpoint="${3:-$SFT_OUTPUT}"
    require_workspace_storage SFT_CHECKPOINT "$checkpoint"
    revision="${BASE_MODEL_REVISION:-main}"
    save_config "$checkpoint" "$model" "$revision"
    load_config
    mkdir -p outputs/server_preflight
    python3 scripts/inspect_server.py --out outputs/server_preflight/environment_before_setup.json
    bash scripts/setup_server.sh
    activate_environment
    python -m unittest discover -s rebuttal/tests -v
    python -m rebuttal.train_sft \
      --base_model "$BASE_MODEL" \
      --model_revision "$BASE_MODEL_REVISION" \
      --train_file "$SFT_TRAIN_FILE" \
      --output_dir "$SFT_CHECKPOINT"
    python scripts/preflight_server.py --checkpoint "$SFT_CHECKPOINT" --out outputs/server_preflight/preflight.json
    echo "SFT bootstrap complete. Next: bash scripts/server_pipeline.sh smoke"
    ;;
  start)
    model="${2:-$BASE_MODEL}"
    checkpoint="${3:-$SFT_OUTPUT}"
    bash scripts/server_pipeline.sh bootstrap-sft "$model" "$checkpoint"
    bash scripts/server_pipeline.sh smoke
    bash scripts/server_pipeline.sh launch-full
    ;;
  smoke)
    require_checkpoint
    activate_environment
    python scripts/preflight_server.py --checkpoint "$SFT_CHECKPOINT"
    python scripts/run_train_matrix.py \
      --checkpoint "$SFT_CHECKPOINT" \
      --base_model "$BASE_MODEL" \
      --suite p0 \
      --seeds 414 \
      --max_samples 8 \
      --output_root outputs/smoke_taxonomy/checkpoints \
      --resume \
      --execute
    echo "Smoke run complete. Next: bash scripts/server_pipeline.sh full"
    ;;
  full)
    require_checkpoint
    activate_environment
    python scripts/preflight_server.py --checkpoint "$SFT_CHECKPOINT"
    python scripts/run_train_matrix.py \
      --checkpoint "$SFT_CHECKPOINT" \
      --base_model "$BASE_MODEL" \
      --suite "$EXPERIMENT_SUITE" \
      --seeds "$SEEDS" \
      --output_root "$RUN_ROOT/checkpoints" \
      --resume \
      --execute
    python scripts/run_eval_matrix.py \
      --checkpoint_root "$RUN_ROOT/checkpoints" \
      --output_root "$RUN_ROOT/diagnostics" \
      --tokenizer "$BASE_MODEL" \
      --suite "$EXPERIMENT_SUITE" \
      --seeds "$SEEDS" \
      --split validation \
      --resume \
      --execute
    python scripts/summarize_server_runs.py --root "$RUN_ROOT"
    echo "Training and validation complete. Review validation before running the test command."
    ;;
  launch-full)
    require_checkpoint
    activate_environment
    mkdir -p outputs/server_logs
    log_file="outputs/server_logs/full_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup bash scripts/server_pipeline.sh full > "$log_file" 2>&1 < /dev/null &
    pid="$!"
    printf '%s\n' "$pid" > outputs/server_logs/full.pid
    printf '%s\n' "$log_file" > outputs/server_logs/latest_full_log.txt
    echo "Started PID $pid"
    echo "Log: $log_file"
    ;;
  ident)
    require_checkpoint
    activate_environment
    python scripts/run_identifiability_matrix.py \
      --checkpoint "$SFT_CHECKPOINT" \
      --seeds "$SEEDS" \
      --output_root outputs/rebuttal_identifiability/checkpoints \
      --resume \
      --execute
    python scripts/summarize_server_runs.py --root outputs/rebuttal_identifiability
    ;;
  status)
    activate_environment
    df -h / "$REPO_ROOT"
    nvidia-smi
    python scripts/summarize_server_runs.py --root outputs
    if [[ -f outputs/server_logs/latest_full_log.txt ]]; then
      log_file="$(<outputs/server_logs/latest_full_log.txt)"
      if [[ -f "$log_file" ]]; then
        echo "----- latest full-run log -----"
        tail -n 40 "$log_file"
      fi
    fi
    ;;
  test)
    activate_environment
    python scripts/run_eval_matrix.py \
      --checkpoint_root "$RUN_ROOT/checkpoints" \
      --output_root "$RUN_ROOT/diagnostics" \
      --tokenizer "$BASE_MODEL" \
      --suite "$EXPERIMENT_SUITE" \
      --seeds "$SEEDS" \
      --split test \
      --resume \
      --execute
    python scripts/summarize_server_runs.py --root "$RUN_ROOT"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
