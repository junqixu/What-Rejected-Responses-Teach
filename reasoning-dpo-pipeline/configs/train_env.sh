# Experiment Configuration — replace paths with your local environment

# ============================================================
# Stage 1: DAG Preference Data Generation
# ============================================================
# Input: path to your raw solution dataset (.json or .jsonl)
INPUT_SOLUTIONS=./data/half/train.json

# Output directory for DAG-generated preference data
DAG_OUTPUT_DIR=./outputs/stage1_dag

# Number of samples per corruption type (computer_error, dependency_mismatch, missing_nodes, disorder)
NUM_PER_TYPE=500


# ============================================================
# Stage 2: LLM-Assisted Preference Generation
# ============================================================
# Source data for LLM generation
LLM_INPUT_PATH=./data/half/train.json

# Output directory for LLM-generated preference data
LLM_OUTPUT_DIR=./outputs/stage2_llm

# DeepSeek API key (set via environment or replace here)
# export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# Number of samples per error type (0 = all)
LLM_SAMPLE_COUNT=0


# ============================================================
# Stage 3: DPO Training
# ============================================================
# Base model (HuggingFace ID or local path)
BASE_MODEL=Qwen/Qwen2-0.5B

# SFT checkpoint (your fine-tuned base model)
SFT_CHECKPOINT=./checkpoints/qwen2_0.5b_sft/checkpoint-10339

# DPO training data (from Stage 1 or Stage 2)
TRAIN_FILE=./outputs/stage1_dag/dpo_mixed.jsonl

# Output directory for DPO checkpoint
DPO_OUTPUT_DIR=./checkpoints/dpo_full

# Full DPO hyperparameters
FULL_DPO_LR=2e-5
FULL_DPO_BATCH=1
FULL_DPO_GRAD_ACCUM=8
FULL_DPO_EPOCHS=1
FULL_DPO_BETA=0.1

# LoRA DPO hyperparameters
LORA_DPO_LR=1e-5
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05


# ============================================================
# Stage 4: Evaluation
# ============================================================
# Validation data
VAL_DATA=./data/half/validation.json

# Evaluation output directory
EVAL_OUTPUT_DIR=./results/eval

# Tokenizer (usually same as base model)
TOKENIZER=Qwen/Qwen2-0.5B


# ============================================================
# Stage 5: RQ3 Batch Evaluation
# ============================================================
# Base directory for all RQ3 results
RQ3_OUTPUT_BASE=./results/rq3

# Shared evaluation data for all RQ3 models
RQ3_VAL_DATA=./data/half/validation.json
