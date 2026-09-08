# Gate 013 — Layer Normalisation

## Status

`OPEN`

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
- `D >= 1`
- `eps > 0`

## Backward semantics

The implementation must expose correct gradients for:

- input `x`
- scale `gamma`
- bias `beta`

The essential LayerNorm backward rule must be implemented explicitly rather than delegated to `torch.nn.LayerNorm` or `torch.nn.functional.layer_norm`.

## Required attacks

The test suite must cover:

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
13. deliberately broken controls for wrong axis, wrong variance convention, epsilon outside the square root, missing centring, incorrect affine broadcasting or detached affine gradients

## Numerical tolerance

Primary oracle comparisons use `float64` and require `torch.testing.assert_close` with:

- `rtol = 1e-9`
- `atol = 1e-10`

Finite-difference checks use central differences with step `1e-6` and tolerance:

- `rtol = 1e-5`
- `atol = 1e-6`

Any test that requires a wider tolerance must document why rather than silently weakening this contract.

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

## Integration authority

No Transformer integration is permitted while this gate is `OPEN`, `FAIL` or `UNRESOLVED`.

## Results

Not yet executed from this repository.
