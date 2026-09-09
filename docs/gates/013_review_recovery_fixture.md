# Gate 013 — Review-Recovery Fixture v1.0

## Status

`FROZEN BEFORE RECOVERY PATCH`

## Purpose

Convert the valid findings from PR #1 into reproducible attacks before changing the LayerNorm implementation or CI configuration.

The fixture is evidence of what must be survived. It is not evidence that the implementation already satisfies the attacks.

## Source findings

The fixture represents the actual independent PR findings:

- FP16 variance accumulation overflow
- non-finite epsilon acceptance
- dtype-policy ambiguity/mismatch risk
- Python 3.11 advertised but not exercised
- workflow permissions not explicitly least-privilege
- checkout credentials persisted by default
- repository quality/docstring warning

## Frozen attacks

### RR-01 — FP16 accumulation safety

Given same-dtype `float16` input and affine parameters with `x = [0, 1000]`, the implementation must:

- produce finite output
- match PyTorch LayerNorm within the frozen FP16 tolerance
- produce finite input/gamma/beta gradients
- match PyTorch gradients within the frozen FP16 tolerance

This attack establishes whether an accepted FP16 input can overflow during internal statistics accumulation.

### RR-02 — Functional non-finite epsilon

`layer_norm` must reject `NaN`, `+inf` and `-inf` epsilon values with `ValueError`.

### RR-03 — Module non-finite epsilon

`LayerNorm` construction must reject `NaN`, `+inf` and `-inf` epsilon values with `ValueError`.

### RR-04 — Dtype policy

Gate 013 does not claim mixed-precision affine support.

Inputs and affine parameters with different dtypes must be rejected explicitly. The recovery patch must not broaden this policy merely to satisfy a reviewer suggestion.

### RR-05 — Python compatibility

While `pyproject.toml` advertises `requires-python = ">=3.11"`, CI must exercise both Python 3.11 and Python 3.12.

### RR-06 — Workflow token permissions

The test workflow must explicitly declare read-only repository contents permission.

### RR-07 — Checkout credential persistence

The checkout step must set `persist-credentials: false`.

### RR-08 — Repository quality

The repository must meet a frozen minimum 80% docstring coverage across Python function/class definitions in `src/` and `tests/`, using the executable fixture's AST-based measurement.

This represents the quality class flagged by the PR pre-merge review rather than suppressing the external warning.

## Initial expected disposition

Before any recovery patch:

- RR-01: expected FAIL
- RR-02: expected FAIL for `NaN` and `+inf`; `-inf` may already fail through the positive-epsilon check
- RR-03: expected FAIL for `NaN` and `+inf`; `-inf` may already fail through the positive-epsilon check
- RR-04: expected PASS because the current contract has already frozen same-dtype semantics
- RR-05: expected FAIL
- RR-06: expected FAIL
- RR-07: expected FAIL
- RR-08: expected FAIL

A discrepancy between expected and executed disposition must be preserved rather than rewritten.

## Patch authority

Only behaviour demonstrated by these attacks, the original Gate 013 suite, or directly required to make the frozen engineering checks executable may be changed during recovery.

Do not add Transformer integration, new model architecture, training logic, mixed-precision support or unrelated refactors.

## Recovery acceptance

The fixture is satisfied only when every RR attack passes, the original Gate 013 tests remain green, the advertised Python compatibility matrix is green, the independent PyTorch oracle checks are green, repository/security checks are green, and external review leaves no valid unresolved finding inside the claimed support envelope.
