# Gate 013 — Layer Normalisation

## Status

`REOPENED`

Initial candidate passed the frozen 25-test suite. Independent PR review subsequently identified untested contract and engineering failures. Gate reopened.

Integration authority is withdrawn pending recovery and re-verification under the frozen engineering gate sequence.

## Objective

Establish LayerNorm as an isolated primitive before any Transformer integration.

## Contract

For an input tensor `x` with shape `(..., D)`, LayerNorm operates only over the final feature dimension `D`.

For each feature vector:

- `mean = x.mean(dim=-1, keepdim=True)`
- `variance = mean((x - mean)^2)` using population variance (`unbiased=False` semantics)
- `inv_std = 1 / sqrt(variance + eps)`
- `x_hat = (x - mean) * inv_std`
- `y = gamma * x_hat + beta`

### Parameters

- `gamma`: learnable scale, shape `(D,)`
- `beta`: learnable bias, shape `(D,)`
- `eps`: fixed finite strictly positive scalar, default `1e-5`

### Input constraints

- input rank must be at least 1
- final input dimension must equal `D`
- input, `gamma` and `beta` must be floating-point tensors
- input, `gamma` and `beta` must share dtype and device
- mixed-dtype affine parameters are outside the Gate 013 contract and must be rejected explicitly
- `D >= 1`
- `eps` must be finite and `eps > 0`

### Recovery support envelope

The recovery claim is bounded to:

- CPU execution in repository CI
- same-dtype input, `gamma` and `beta`
- explicit attack coverage for `float16`, `float32` and `float64`
- Python versions advertised by `pyproject.toml`
- finite input values within the explicit adversarial cases

GPU-specific behaviour, bfloat16 numerical correctness, higher-order gradients and mixed-precision affine execution remain non-claims for Gate 013 recovery.

## Backward semantics

The implementation must expose correct gradients for:

- input `x`
- scale `gamma`
- bias `beta`

The essential LayerNorm backward rule must be implemented explicitly rather than delegated to `torch.nn.LayerNorm` or `torch.nn.functional.layer_norm`.

## Original required attacks

The initial 25-test suite covered:

1. forward agreement with PyTorch LayerNorm
2. input-gradient agreement with PyTorch
3. `gamma`-gradient agreement with PyTorch
4. `beta`-gradient agreement with PyTorch
5. independent central finite-difference checks
6. sequence-position isolation
7. batch isolation
8. within-vector feature coupling
9. `D=1`
10. constant vectors
11. very small variance
12. large-magnitude numerically safe inputs
13. deliberately broken controls for wrong axis, wrong variance convention, epsilon outside the square root, missing centring, incorrect affine broadcasting and detached affine gradients

These attacks remain part of the gate but are no longer sufficient for PASS.

## Review-Recovery Fixture

The additional frozen attacks are defined in `docs/gates/013_review_recovery_fixture.md` and executed by `tests/test_gate_013_review_recovery.py`.

They cover:

1. FP16 variance accumulation overflow, forward and backward
2. non-finite epsilon rejection in the functional API
3. non-finite epsilon rejection in the module API
4. explicit same-dtype policy for affine parameters
5. Python 3.11/3.12 compatibility coverage while `>=3.11` is advertised
6. explicit read-only GitHub Actions permissions
7. disabled checkout credential persistence
8. repository documentation-quality threshold

## Numerical tolerance

Primary float64 oracle comparisons retain:

- `rtol = 1e-9`
- `atol = 1e-10`

Finite-difference checks retain:

- step `1e-6`
- `rtol = 1e-5`
- `atol = 1e-6`

The FP16 recovery attack uses wider tolerances appropriate to half precision and requires all compared outputs and gradients to remain finite.

## Explicit non-claims

This gate does not establish that LayerNorm:

- improves optimisation
- improves convergence
- improves model quality
- is correctly positioned inside a Transformer block
- preserves causal or autoregressive behaviour after integration
- is numerically verified on GPU
- is numerically verified for bfloat16
- supports mixed-dtype affine parameters
- supports higher-order gradients

## Acceptance criterion

`PASS` iff the original required attacks, the frozen review-recovery fixture, repository quality/security checks, the advertised compatibility matrix, independent oracle checks and clean CI all pass, and external review identifies no unresolved valid defect within the claimed support envelope.

`FAIL` iff a required property is demonstrably violated.

`UNRESOLVED` iff available evidence is insufficient to decide.

No earlier stage in the frozen engineering sequence independently authorises PASS.

## Historical execution evidence

The initial candidate produced this repository evidence before review recovery:

- repository: `9TEVE-O/steven-llm-build`
- PR: `#1` — Gate 013: standalone LayerNorm primitive
- tested head commit: `351a152f831216b7b2ec88bfb5c4ac4b27864275`
- workflow run: `34182203608`
- runner: Ubuntu 24.04
- Python: 3.12.14
- PyTorch: 2.14.0
- pytest: 9.1.1
- result: `25 passed`
- pytest duration: `2.31s`

This evidence is preserved. It establishes that the initial candidate passed the original frozen 25-test suite. It does not establish current Gate 013 PASS.

## Current decision

`REOPENED`

Independent PR review identified valid failures and evidence gaps outside the original 25-test attack surface. Recovery attacks are frozen before implementation changes.

## Integration authority

`WITHDRAWN`

No Transformer integration is authorised by Gate 013 until the reopened gate earns a new PASS under the frozen engineering gate sequence.

## Remaining uncertainty

- GPU-specific numerical behaviour is not tested
- bfloat16 numerical correctness is not established
- higher-order gradients are not tested
- optimisation or training benefit is not tested
- Transformer integration has not been tested
