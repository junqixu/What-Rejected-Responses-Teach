# Rebuttal experiment configs

These `.yaml` files intentionally use JSON syntax, which is valid YAML and lets
`python -m rebuttal.train_dpo --config ...` read them without adding a YAML
dependency. Paths are isolated under `data/rebuttal/` and `outputs/`; each run
must override `--seed` with `414`, `6201`, and `2026` for the full experiment.

The `taxonomy_*` configs consume the compact strict prompt intersection in
`data/rebuttal_release/train.jsonl`. Single-axis configs use the trainer's
`error_types` filter, so no duplicate per-type files are required. `taxonomy_mix_sum` is the main
eight-axis run, `taxonomy_mix_mean` is the response-length control, and the
Operation/Wrong-Target configs are the two prioritized single-axis extensions.
