# Gate 013 — Layer Normalisation

## Status

`PASS`

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
- `eps`: fixed positive scalar, default `1e-5`

### Input constraints

- input rank must be at least 1
- final input dimension must equal `D`
- input, `gamma` and `beta` must be floating-point tensors
- input, `gamma` and `beta` must share dtype and device
- `D >= 1`
- `eps > 0`

## Backward semantics

The implementation must expose correct gradients for:

- input `x`
- scale `gamma`
- bias `beta`

The essential LayerNorm backward rule must be implemented explicitly rather than delegated to `torch.nn.LayerNorm` or `torch.nn.functional.layer_norm`.

## Required attacks

The test suite covers:

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

## Numerical tolerance

Primary oracle comparisons use `float64` and require `torch.testing.assert_close` with:

- `rtol = 1e-9`
- `atol = 1e-10`

Finite-difference checks use central differences with step `1e-6` and tolerance:

- `rtol = 1e-5`
- `atol = 1e-6`

No test required a wider tolerance.

## Explicit non-claims

This gate does not establish that LayerNorm:

- improves optimisation
- improves convergence
- improves model quality
- is correctly positioned inside a Transformer block
- preserves causal or autoregressive behaviour after integration

Those are integration or later-gate questions.

## Acceptance criterion

`PASS` iff all required standalone tests pass and the negative controls are detected.

`FAIL` iff a required property is demonstrably violated.

`UNRESOLVED` iff available evidence is insufficient to decide.

## Execution evidence

GitHub Actions PR run:

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

One warning was emitted because PyTorch attempted optional NumPy initialisation while NumPy was not installed. No Gate 013 test depends on NumPy and the warning did not alter test outcomes.

A separate local diagnostic of the explicit backward equations also produced maximum PyTorch-oracle gradient differences on the order of `1e-16` and finite-difference discrepancies on the order of `1e-10`; this is supplementary only. The GitHub Actions run is the repository execution evidence.

## Decision

`PASS`

Within the frozen tested domain, the standalone LayerNorm primitive satisfies the Gate 013 contract. The suite also detects the specified deliberately broken controls.

This PASS establishes only the isolated primitive. It does not establish Transformer integration behaviour.

## Integration authority

Standalone LayerNorm integration is now permitted by Gate 013, but no Transformer integration is included in this gate or PR.

A separate integration gate must freeze LayerNorm placement and rerun the affected causal and autoregressive regression properties before those integrated claims receive support.

## Remaining uncertainty

- behaviour outside the tested dtype/device/input domain is not established
- GPU-specific numerical behaviour was not tested
- higher-order gradients were not tested
- optimisation or training benefit was not tested
- Transformer integration has not been tested
