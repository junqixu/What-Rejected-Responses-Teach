# DeepSeek eight-type dataset reconstruction

## Current state

`data/rebuttal_v2/source/` contains clean OP=10 source pools prepared from all
three legacy `data/half/` files. Splitting is based on the legacy `id` as a DAG
family key and prompt-content hashes as prompt IDs. The selected pools contain
1,000 train, 200 validation, and 500 test prompts with zero prompt or family
overlap.

The first DeepSeek pass is preserved under `data/rebuttal_v2/deepseek/`. It must
not be used for formal training because its Spurious-Node definition allowed an
unused dead node. The strict, versioned dataset is under
`data/rebuttal_v3/deepseek/`: seven valid v2 axes were curated into it and the
Spurious-Node axis was rebuilt with deterministic downstream-used graph edits.

The legacy data remain under `data/dpo_outputs/` and `data/rebuttal/`. They are
not overwritten by this workflow.

## Credential setup

Do not commit or paste the key into source files. Configure it only in the shell
that launches generation:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
python -m pip install openai
```

Linux:

```bash
export DEEPSEEK_API_KEY="your-key"
python -m pip install openai
```

## API smoke test

```bash
python -m rebuttal.generation.deepseek_eight_types \
  --input data/rebuttal_v2/source/train_clean.jsonl \
  --out data/rebuttal_v2/deepseek_smoke/train \
  --split train \
  --max_samples 5 \
  --max_api_calls 5 \
  --max_retries 1
```

Inspect `dataset_manifest.json`, `generation_failures.jsonl`, and every per-type
JSONL before increasing the sample count.

## Full generation

Run each split into a separate v3 directory. The generator resumes by prompt ID and
error type, so rerunning the same command requests only missing records.

```bash
python -m rebuttal.generation.deepseek_eight_types \
  --input data/rebuttal_v2/source/train_clean.jsonl \
  --out data/rebuttal_v3/deepseek/train \
  --split train \
  --max_api_calls 1200 \
  --max_retries 3

python -m rebuttal.generation.deepseek_eight_types \
  --input data/rebuttal_v2/source/validation_clean.jsonl \
  --out data/rebuttal_v3/deepseek/validation \
  --split validation \
  --max_api_calls 300 \
  --max_retries 3

python -m rebuttal.generation.deepseek_eight_types \
  --input data/rebuttal_v2/source/test_clean.jsonl \
  --out data/rebuttal_v3/deepseek/test \
  --split test \
  --max_api_calls 650 \
  --max_retries 3
```

One primary call asks for all eight types for one prompt. Validation failures are
retried only for the remaining types, up to the configured limit. API-call caps
are safety limits, not expected final counts.

## Post-generation validation

```bash
python -m rebuttal.taxonomy.validate_perturbations \
  --data data/rebuttal_v3/deepseek/train/balanced/all_types.jsonl \
  --out outputs/rebuttal_taxonomy/deepseek_train_validation
```

Do not train until all eight per-type counts, complete-prompt count, hash checks,
prompt/family leakage checks, and failure reasons have been reviewed.

For an existing v2 directory, migrate without overwriting it and rebuild the
strict Spurious-Node axis with:

```bash
python -m rebuttal.generation.curate_deepseek_dataset \
  --source_dir data/rebuttal_v2/deepseek/train \
  --out data/rebuttal_v3/deepseek/train \
  --split train

python -m rebuttal.generation.deterministic_backfill \
  --input data/rebuttal_v2/source/train_clean.jsonl \
  --out data/rebuttal_v3/deepseek/train \
  --split train \
  --error_types spurious_node
```
