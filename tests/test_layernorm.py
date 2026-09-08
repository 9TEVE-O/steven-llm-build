from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from steven_llm.nn.layernorm import LayerNorm, layer_norm


RTOL = 1e-9
ATOL = 1e-10
FD_RTOL = 1e-5
FD_ATOL = 1e-6
EPS = 1e-5


def _manual_reference(x, gamma, beta, eps=EPS):
    """Compute the direct LayerNorm reference equations for finite differences."""
    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    variance = centred.square().mean(dim=-1, keepdim=True)
    return centred / torch.sqrt(variance + eps) * gamma + beta


def _finite_difference(fn, value, step=1e-6):
    """Estimate a gradient with independent central finite differences."""
    grad = torch.empty_like(value)
    flat_grad = grad.reshape(-1)
    for i in range(value.numel()):
        plus = value.clone()
        minus = value.clone()
        plus.reshape(-1)[i] += step
        minus.reshape(-1)[i] -= step
        flat_grad[i] = (fn(plus) - fn(minus)) / (2 * step)
    return grad


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_forward_matches_pytorch(shape):
    """Match PyTorch LayerNorm forward values across supported tensor ranks."""
    torch.manual_seed(13)
    x = torch.randn(shape, dtype=torch.float64)
    gamma = torch.randn(shape[-1], dtype=torch.float64)
    beta = torch.randn(shape[-1], dtype=torch.float64)

    actual = layer_norm(x, gamma, beta, EPS)
    expected = F.layer_norm(x, (shape[-1],), gamma, beta, EPS)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_backward_matches_pytorch_for_input_gamma_and_beta():
    """Match PyTorch gradients for input and both affine parameters."""
    torch.manual_seed(1301)
    shape = (2, 3, 5)
    upstream = torch.randn(shape, dtype=torch.float64)

    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
    beta = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
    (layer_norm(x, gamma, beta, EPS) * upstream).sum().backward()
    actual_grads = (x.grad.clone(), gamma.grad.clone(), beta.grad.clone())

    x_ref = x.detach().clone().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    (F.layer_norm(x_ref, (shape[-1],), gamma_ref, beta_ref, EPS) * upstream).sum().backward()
    expected_grads = (x_ref.grad, gamma_ref.grad, beta_ref.grad)

    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_gradients_match_independent_central_finite_differences():
    """Match independently estimated gradients for input, gamma and beta."""
    torch.manual_seed(1302)
    x0 = torch.randn((2, 3), dtype=torch.float64)
    gamma0 = torch.randn(3, dtype=torch.float64)
    beta0 = torch.randn(3, dtype=torch.float64)
    upstream = torch.randn((2, 3), dtype=torch.float64)

    x = x0.clone().requires_grad_(True)
    gamma = gamma0.clone().requires_grad_(True)
    beta = beta0.clone().requires_grad_(True)
    loss = (layer_norm(x, gamma, beta, EPS) * upstream).sum()
    loss.backward()

    fd_x = _finite_difference(
        lambda candidate: (_manual_reference(candidate, gamma0, beta0) * upstream).sum(), x0
    )
    fd_gamma = _finite_difference(
        lambda candidate: (_manual_reference(x0, candidate, beta0) * upstream).sum(), gamma0
    )
    fd_beta = _finite_difference(
        lambda candidate: (_manual_reference(x0, gamma0, candidate) * upstream).sum(), beta0
    )

    torch.testing.assert_close(x.grad, fd_x, rtol=FD_RTOL, atol=FD_ATOL)
    torch.testing.assert_close(gamma.grad, fd_gamma, rtol=FD_RTOL, atol=FD_ATOL)
    torch.testing.assert_close(beta.grad, fd_beta, rtol=FD_RTOL, atol=FD_ATOL)


def test_sequence_position_isolation():
    """Changing one sequence position must not alter another position's result."""
    x = torch.tensor(
        [[[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0], [4.0, 6.0, 7.0, 11.0]]],
        dtype=torch.float64,
    )
    gamma = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    beta = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)

    baseline = layer_norm(x, gamma, beta)
    changed = x.clone()
    changed[0, 2] += torch.tensor([100.0, -50.0, 20.0, 7.0], dtype=torch.float64)
    mutated = layer_norm(changed, gamma, beta)

    torch.testing.assert_close(baseline[0, 1], mutated[0, 1], rtol=0.0, atol=0.0)


def test_batch_isolation():
    """Changing one batch item must not alter another batch item's result."""
    x = torch.tensor(
        [[[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]], [[10.0, 20.0, 40.0], [30.0, 50.0, 80.0]]],
        dtype=torch.float64,
    )
    gamma = torch.ones(3, dtype=torch.float64)
    beta = torch.zeros(3, dtype=torch.float64)

    baseline = layer_norm(x, gamma, beta)
    changed = x.clone()
    changed[1] *= -17.0
    mutated = layer_norm(changed, gamma, beta)

    torch.testing.assert_close(baseline[0], mutated[0], rtol=0.0, atol=0.0)


def test_features_are_coupled_within_one_normalised_vector():
    """Changing one feature must affect peer features within the same normalised vector."""
    x = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
    gamma = torch.ones(4, dtype=torch.float64)
    beta = torch.zeros(4, dtype=torch.float64)

    baseline = layer_norm(x, gamma, beta)
    changed = x.clone()
    changed[0, 0] += 10.0
    mutated = layer_norm(changed, gamma, beta)

    assert not torch.allclose(baseline[0, 1:], mutated[0, 1:])


def test_feature_dimension_one_is_finite_and_has_expected_gradients():
    """Handle the D=1 degenerate case with finite output and exact gradients."""
    x = torch.tensor([[[2.0]], [[-7.0]]], dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([1.25], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([[[2.0]], [[-5.0]]], dtype=torch.float64)

    output = layer_norm(x, gamma, beta)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.full_like(output, 1.25), rtol=0.0, atol=0.0)

    (output * upstream).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma.grad, torch.zeros_like(gamma), rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta.grad, upstream.sum(dim=(0, 1)), rtol=0.0, atol=0.0)


def test_constant_nonzero_vectors_are_finite_and_reduce_to_beta():
    """Constant vectors must normalise to zero and therefore reduce to beta."""
    x = torch.full((2, 3, 4), 7.5, dtype=torch.float64)
    gamma = torch.tensor([0.5, -1.0, 2.0, 3.0], dtype=torch.float64)
    beta = torch.tensor([-3.0, -1.0, 2.0, 8.0], dtype=torch.float64)

    output = layer_norm(x, gamma, beta)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, beta.expand_as(output), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 1.0 + 1e-10, 1.0 - 1e-10, 1.0 + 2e-10]], dtype=torch.float64),
        torch.tensor([[1e8, 1e8 + 1.0, 1e8 - 2.0, 1e8 + 4.0]], dtype=torch.float64),
    ],
)
def test_small_variance_and_large_safe_magnitudes_match_pytorch(x):
    """Match PyTorch for small variance and large float64 magnitudes that remain numerically safe."""
    gamma = torch.tensor([1.0, -0.5, 2.0, 0.25], dtype=torch.float64)
    beta = torch.tensor([0.0, 1.0, -1.0, 3.0], dtype=torch.float64)

    actual = layer_norm(x, gamma, beta)
    expected = F.layer_norm(x, (4,), gamma, beta, EPS)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_module_initialises_identity_affine_parameters():
    """Initialise gamma to one and beta to zero."""
    module = LayerNorm(4).double()
    torch.testing.assert_close(module.gamma, torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(module.beta, torch.zeros(4, dtype=torch.float64))


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda: LayerNorm(0), ValueError),
        (lambda: LayerNorm(3, eps=0.0), ValueError),
        (lambda: layer_norm(torch.tensor(1.0), torch.ones(1), torch.zeros(1)), ValueError),
        (lambda: layer_norm(torch.ones(2, 3), torch.ones(4), torch.zeros(4)), ValueError),
        (lambda: layer_norm(torch.ones(2, 3, dtype=torch.int64), torch.ones(3), torch.zeros(3)), TypeError),
        (lambda: layer_norm(torch.ones(2, 3, dtype=torch.float32), torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)), ValueError),
        (lambda: layer_norm(torch.ones(2, 3), torch.ones(3), torch.zeros(3), 0.0), ValueError),
    ],
)
def test_invalid_contract_inputs_fail_explicitly(call, error):
    """Reject each frozen invalid-input class with the specified exception type."""
    with pytest.raises(error):
        call()


def _negative_control_fixture():
    """Return a non-symmetric fixture capable of exposing defective LayerNorm variants."""
    x = torch.tensor(
        [
            [[1.0, 2.0, 4.0, 8.0], [2.0, 5.0, 9.0, 15.0], [3.0, 7.0, 10.0, 21.0]],
            [[-2.0, 1.0, 5.0, 12.0], [4.0, 6.0, 13.0, 20.0], [8.0, 11.0, 17.0, 29.0]],
        ],
        dtype=torch.float64,
    )
    gamma = torch.tensor([0.7, -1.1, 2.0, 0.5], dtype=torch.float64)
    beta = torch.tensor([-0.2, 0.4, 1.3, -2.0], dtype=torch.float64)
    return x, gamma, beta


def test_negative_control_wrong_axis_is_detected():
    """Detect a mutant that normalises over the wrong axis."""
    x, gamma, beta = _negative_control_fixture()
    correct = layer_norm(x, gamma, beta)
    mean = x.mean(dim=-2, keepdim=True)
    variance = (x - mean).square().mean(dim=-2, keepdim=True)
    mutant = (x - mean) / torch.sqrt(variance + EPS) * gamma + beta
    assert not torch.allclose(correct, mutant, rtol=RTOL, atol=ATOL)


def test_negative_control_unbiased_variance_is_detected():
    """Detect a mutant that uses sample rather than population variance."""
    x, gamma, beta = _negative_control_fixture()
    correct = layer_norm(x, gamma, beta)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=True)
    mutant = (x - mean) / torch.sqrt(variance + EPS) * gamma + beta
    assert not torch.allclose(correct, mutant, rtol=RTOL, atol=ATOL)


def test_negative_control_epsilon_outside_sqrt_is_detected():
    """Detect a mutant that places epsilon outside the square root."""
    x, gamma, beta = _negative_control_fixture()
    correct = layer_norm(x, gamma, beta)
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    mutant = (x - mean) / (torch.sqrt(variance) + EPS) * gamma + beta
    assert not torch.allclose(correct, mutant, rtol=RTOL, atol=ATOL)


def test_negative_control_missing_centring_is_detected():
    """Detect a mutant that omits mean centring."""
    x, gamma, beta = _negative_control_fixture()
    correct = layer_norm(x, gamma, beta)
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    mutant = x / torch.sqrt(variance + EPS) * gamma + beta
    assert not torch.allclose(correct, mutant, rtol=RTOL, atol=ATOL)


def test_negative_control_wrong_affine_broadcast_is_detected():
    """Detect a mutant that collapses affine parameters before broadcasting."""
    x, gamma, beta = _negative_control_fixture()
    correct = layer_norm(x, gamma, beta)
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(variance + EPS)
    mutant = x_hat * gamma.mean() + beta.mean()
    assert not torch.allclose(correct, mutant, rtol=RTOL, atol=ATOL)


def test_negative_control_detached_gamma_gradient_is_detected():
    """Detect a mutant that detaches gamma from gradient flow."""
    x = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).square().mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(variance + EPS)
    mutant = x_hat * gamma.detach() + beta
    mutant.sum().backward()
    assert gamma.grad is None
