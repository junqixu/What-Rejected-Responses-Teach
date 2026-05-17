# What-Rejected-Responses-Teach

![Project Overview](overall.pdf])
A complete, research-ready codebase for training and evaluating large language models
using Direct Preference Optimisation (DPO) on structured math reasoning tasks.

The pipeline generates preference data at scale using **two complementary approaches**,
trains DPO models (full-parameter or LoRA), and evaluates them with multi-level
reasoning-chain metrics that go beyond simple answer correctness.

---

## Project Structure

```
reasoning-dpo-pipeline/
│
├── stage1_dag_preference/          # Rule-based DAG preference data
│   ├── dag_parser.py              # Parse solutions → DAG
│   ├── dag_corruptor.py          # Corrupt DAG → rejected response
│   └── run_pipeline.py           # End-to-end pipeline
│
├── stage2_llm_generation/         # LLM-assisted preference data
│   └── run_llm_preference.py      # DeepSeek / OpenAI API generator
│
├── stage3_training/               # DPO training
│   ├── trainer_base.py            # Shared utilities (tokenizer, data, model)
│   ├── full_dpo/
│   │   └── run_full_dpo.py       # Full-parameter fine-tuning
│   └── lora_dpo/
│       └── run_lora_dpo.py       # LoRA fine-tuning
│
├── stage4_evaluation/             # Evaluation engine
│   ├── run_eval.py               # vLLM + reasoning-chain metrics
│   └── analysis.py               # Load & normalise results → CSV
│
├── stage5_rq3_experiments/       # Batch evaluation for RQ1–RQ3
│   └── batch_eval.py             # Multi-model eval + report generation
│
├── results/
│   ├── generate_figures.py       # Thesis figure generator
│   ├── sample_outputs/           # Example evaluation outputs
│   └── figures/                  # Generated figures (gitignored)
│
├── configs/
│   ├── default_config.json        # All hyperparameters (JSON)
│   └── train_env.sh              # Shell environment variables
│
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                        # Unit & integration tests
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

---

## Overview of the Research Pipeline

```
Raw Solutions
    │
    ├──────────────────────────────────────────────┐
    │  Stage 1: Rule-based (DAG)                 │ Stage 2: LLM-assisted
    │  ─────────────────────────────              │ ───────────────────
    │  Parse solution → DAG                       │ DeepSeek API call
    │  Apply 4 corruption strategies:              │ Prompt → rejected
    │    • computer_error                        │ response with:
    │    • dependency_mismatch                   │   • computation_error
    │    • missing_nodes                        │   • dependency_mismatch
    │    • disorder                             │   • missing_nodes
    │  QC → (prompt, chosen, rejected)          │ QC → preference pair
    └──────────────────────────────────────────────┘
                            │
                            ▼
                    Preference Dataset
                    (DAG + LLM mixed)
                            │
                    ┌───────┴────────┐
                    │                │
              Stage 3a           Stage 3b
            Full DPO           LoRA DPO
           (Qwen2-0.5B)      (Qwen2-0.5B)
                │                  │
                └────────┬─────────┘
                         ▼
               Trained DPO Checkpoints
                │         │         │
           Exp1     Exp2    Exp3
         (pair mix) (inj.)  (other)
                         │
                         ▼
               Stage 4: vLLM Evaluation
               ──────────────────────────
               k candidates per problem
               Metrics:
                 • answer-pass@k
                 • reasoning-chain-pass@k
                 • answer-reasoning gap
                 • generalisation-decay
                         │
                         ▼
               Stage 5: Batch RQ3 Reports
               ────────────────────────────
               Markdown + CSV comparison
               across all experiment groups
                         │
                         ▼
               Thesis Figures (matplotlib)
```

---

## Core Concepts

### DAG-Based Corruption (Stage 1)

A "solution" is a structured chain-of-thought text with `Define X as ...;` steps.
The `DAGParser` extracts a directed graph where each node is one reasoning step
and edges encode variable-level dependencies.

The `DAGCorruptor` introduces four types of reasoning errors at the **structural** level:

| Error Type | What it does |
|---|---|
| `computer_error` | Perturbs one intermediate number (±1 or ±2) |
| `dependency_mismatch` | Replaces one variable reference with a wrong one |
| `missing_nodes` | Deletes 1–2 critical intermediate steps |
| `disorder` | Swaps the order of two causally dependent steps |

### Reasoning Chain Metrics (Stage 4)

Standard `answer-pass@k` only measures whether the final number is correct.
We introduce **reasoning-chain-pass@k**, which measures whether the generated
text contains a structurally valid dependency graph:

```
graph_score = 0.45 × edge_F1 + 0.20 × node_F1
            + 0.20 × entity_F1 + 0.15 × raw_edge_F1
equation_score = F1 of extracted equation steps

chain_score = max(graph_score, equation_score)
chain_correct = (answer_correct AND chain_score >= 0.35)
```

The **answer-reasoning gap** (answer-pass@128 − reasoning-chain-pass@128)
quantifies how often the model arrives at the correct answer by luck rather
than genuine reasoning.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Edit configs/train_env.sh with your paths
source configs/train_env.sh

# Set API key for Stage 2
export DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. Run the full pipeline

```bash
# Stage 1 — generate DAG-based preference data
python -m stage1_dag_preference.run_pipeline \
    --input_path ./data/half/train.json \
    --output_dir ./outputs/stage1_dag \
    --num_per_type 500

# Stage 3 — full DPO training
python -m stage3_training.full_dpo.run_full_dpo \
    --sft_checkpoint ./checkpoints/qwen2_0.5b_sft/checkpoint-10339 \
    --train_file ./outputs/stage1_dag/dpo_mixed.jsonl \
    --output_dir ./checkpoints/dpo_full

# Stage 4 — evaluate the model
python -m stage4_evaluation.vllm_eval.run_eval \
    --model_path ./checkpoints/dpo_full \
    --data_path ./data/half/validation.json \
    --output_dir ./results/eval_dpo_full

# Stage 5 — RQ3 batch evaluation
python -m stage5_rq3_experiments.batch_eval \
    --exp all \
    --output_base ./results/rq3

# Generate thesis figures
python -m results.generate_figures \
    --input_csv ./results/tables/merged_all_details.csv \
    --output_dir ./results/thesis_figures
```

---

## Training Hardware

| Script | GPU | VRAM | Notes |
|---|---|---|---|
| `full_dpo` | RTX 4090 / A100 | ~20 GB | bf16, gradient checkpointing |
| `lora_dpo` | RTX 3090 / A100 | ~12 GB | rank 16 LoRA |

---

## Data Format

### Input (raw solutions)

```json
[
  {
    "id": "sample_001",
    "problem": "Alice has 3 apples...",
    "question": "How many apples does Alice have left?",
    "solution": "Define a as 3; Define b as 2; so c = a - b; Answer: 1",
    "op": 4,
    "template": "basic_arithmetic"
  }
]
```

### Output (DPO preference)

```json
[
  {
    "pair_id": "sample_001__disorder__0",
    "id": "sample_001",
    "error_type": "disorder",
    "prompt": "<|im_start|>user\nProblem:\nAlice has 3 apples...\n\n...",
    "chosen": "Define a as 3; Define b as 2; so c = a - b; Answer: 1",
    "rejected": "Define b as 2; Define a as 3; so c = a - b; Answer: 1",
    "dag": { "preamble": "", "order": ["a", "b", "c"], ... }
  }
]
```

---

## Config Files

| File | Purpose |
|---|---|
| `configs/default_config.json` | All hyperparameters in one JSON file |
| `configs/train_env.sh` | Shell environment variables (paths, API keys) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{reasoning_dpo_pipeline,
  title={Reasoning-DPO Pipeline},
  author={},
  year={2026},
  url={https://github.com/your-org/reasoning-dpo-pipeline}
}
```
