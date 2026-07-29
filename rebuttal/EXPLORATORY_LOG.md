# Exploratory log

No result-guided thresholds, data rules, or evaluation rules have been changed.

2026-07-26: The first Computation + Dependency construction retained different
prompt pools across comparative regimes because computation edits were not
applicable to 32 traces. Before any model training or evaluation, construction
was corrected to require A, B, and A+B eligibility for every comparative regime.
Single-axis files intentionally retain their independently eligible pools. This
change was driven by the preregistered same-prompt-pool requirement, not results.
