# Rebuttal implementation plan

## Principles

Keep existing experiments immutable, write only to `data/rebuttal/` and
`outputs/rebuttal_*`, hold model initialization and optimization fixed across
regimes, group splits by prompt ID, preserve failures, and never fill an absent
metric with an inferred or fabricated value.

## P0 phases

1. **Audit and contracts — complete.** Record existing paths, schemas, settings,
   missing components, and checkpoint assumptions.
2. **Identifiability tooling — complete offline.** Canonicalize eight structural
   axes; generate delta/exposure matrices; save SVD, effective rank, condition
   number, correlations, variances, and nullspace basis.
3. **Controlled data construction — complete offline.** Generate Confounded,
   Orthogonal-Size, Orthogonal-Exposure, Factorial, and single-axis regimes from
   one eligible prompt pool. Preserve per-record hashes and edit metadata.
4. **Surface controls — complete offline.** Produce token/length/edit/step
   statistics, grouped metadata-only classification, and exact-bucket plus
   nearest-neighbor matching with standardized mean differences.
5. **DPO objective control — code complete, GPU pending.** Train standard `sum`
   and token-mean `mean` runs from the same checkpoint/config. First run
   `--dry_run --max_samples 8`; then execute three seeds only in the verified GPU
   environment.
6. **Diagnostic evaluation — pending external assets.** Implement scored
   clean-vs-A, clean-vs-B, and clean-vs-A+B evaluation after checkpoint and TRL
   interfaces are confirmed. Keep raw and length-normalized margins.
7. **Answer/Reasoning evaluation — partly blocked.** Reuse legacy Answer@k only
   for answer correctness. Obtain the actual reasoning evaluator before reporting
   Reasoning@128.
8. **Aggregation/statistics — partial.** Design-only aggregation is implemented.
   Add paired bootstrap, exact McNemar, Holm correction, and effect sizes once
   paired prediction files exist.
9. **Taxonomy/natural errors — controlled edits complete; natural-error study
   pending.** The annotation queue must follow model generation and automatic
   trace-to-gold comparison; do not infer coverage from controlled negatives.

## Required execution order

For each pair and seed: generate data, verify prompt-pool identity and matrix
rank, dry-run both trainer reductions, train from the same SFT checkpoint, score
the same held-out pair files, run Answer@128 and the verified Reasoning@128
evaluator, aggregate raw predictions, then freeze the result manifest. Any
threshold or data-rule change after viewing results belongs in
`rebuttal/EXPLORATORY_LOG.md` and requires a new output namespace.
