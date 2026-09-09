# Gate 013 — Pre-Review Fixture v1.0

## Status

`FROZEN AFTER RECOVERY PATCH / BEFORE EXTERNAL REVIEW`

## Purpose

Represent criticism classes that a capable independent reviewer could reasonably apply before Gate 013 is ever reconsidered for PASS.

This fixture does not replace the frozen review-recovery fixture. It adds attacks discovered during the required internal pre-review stage.

## PR-01 — Default float32 oracle

The module defaults to float32 parameters, so float32 is part of the practical support surface. Forward values and input/gamma/beta gradients must match PyTorch LayerNorm within float32-appropriate tolerances.

## PR-02 — Independent gradcheck

The explicit backward implementation must pass PyTorch `gradcheck` in float64. This supplements, rather than replaces, the existing hand-written central finite-difference test.

## PR-03 — Extreme finite FP16 range

For the largest finite signed float16 values, `[-65504, 65504]`, LayerNorm output must remain finite and match the expected normalised direction. This attacks the same accumulation-overflow mechanism at the finite dtype boundary rather than only the original `[0, 1000]` reproducer.

## Decision rule

Failure of any pre-review attack prevents Gate 013 PASS. A failing attack must be preserved and patched locally before the applicable engineering sequence is rerun.
