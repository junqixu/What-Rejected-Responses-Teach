# Rebuttal experiment status

Updated: 2026-07-30.

## Release-ready offline artifacts

- The frozen eight-axis release lives in `data/rebuttal_release/` and passes the
  strict structural validator with no failures.
- Train: 540 prompts, 4,320 preference records, 402 DAG families.
- Validation: 85 prompts, 680 preference records, 62 DAG families.
- Test: 203 prompts, 1,624 preference records, 78 DAG families.
- Every prompt has exactly one valid record for each of the eight error axes.
  Prompt overlap and DAG-family overlap across all splits are both zero.
- The release manifest records source composition, generation models/seeds,
  per-file SHA256 hashes, error counts, family counts, and leakage checks.
- DeepSeek-generated records were curated with type-specific structural checks;
  deterministic graph edits fill only validated gaps. All 972 legacy
  Spurious-Node records were excluded because their inserted nodes were unused.
- The replacement Spurious-Node construction requires the inserted value to be
  consumed downstream, and this invariant is covered by unit tests.

## Code and reproducibility

- Server setup, data preflight, taxonomy and identifiability training matrices,
  held-out evaluation, Answer@K evaluation, aggregation, and release regeneration
  have command-line entry points documented in the repository README.
- Training preflight rejects duplicate IDs, invalid response hashes, inconsistent
  chosen traces, incomplete error coverage, and manifest hash mismatches.
- Each training run records data/config/manifest hashes, Git commit, environment,
  seed, checkpoint, and completion or failure status.
- Sum-log-probability and token-mean DPO objectives share one implementation and
  can be compared with otherwise fixed settings.
- All 28 offline unit tests pass. Data preflight, bytecode compilation, the P0
  taxonomy matrix, the held-out evaluation matrix, and all identifiability
  conditions pass dry-run validation from the repository root.
- The A800 server workflow now has a secret-free environment inspector, a single
  pipeline entry point, background launch/status commands, provenance-aware
  resume behavior, and a Chinese step-by-step runbook.

## Existing supporting experiments

- Missing-Node + Disorder and Computation + Dependency datasets include
  confounded, orthogonal-size, orthogonal-exposure, and factorial conditions.
- Confounded designs have observed rank 1; all orthogonal and factorial designs
  have rank 2. The eight-axis release design has rank 8 and condition number 1.0.
- The grouped surface-metadata audit reached accuracy 0.8758 and macro-F1 0.8298,
  demonstrating a control concern rather than a DPO performance result.
- Computation/Missing matching produced only 35 pairs with residual imbalance and
  is explicitly marked as a failed control, not treated as evidence.

## Requires the GPU server

- A local or Hugging Face SFT checkpoint, exposed as `SFT_CHECKPOINT`.
- CUDA-capable PyTorch and the pinned training dependencies.
- GPU smoke training followed by the full three-seed taxonomy and
  identifiability matrices.
- Answer@128, verified reasoning evaluation, location/topology OOD evaluation,
  gradient geometry, paired bootstrap/McNemar/Holm tests, and final plots from
  model prediction files.

No full GPU training has been claimed or fabricated. API keys, `.env` files,
checkpoints, caches, and generated training outputs are excluded from Git.
