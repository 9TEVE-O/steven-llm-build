"""Executable recovery attacks derived from PR #1 review findings."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from steven_llm.nn.layernorm import LayerNorm, layer_norm


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src"
TESTS = ROOT / "tests"
EPS = 1e-5


def test_rr01_float16_large_range_forward_and_backward_match_pytorch() -> None:
    """Accepted float16 inputs must not overflow during variance accumulation."""
    x = torch.tensor([[0.0, 1000.0]], dtype=torch.float16, requires_grad=True)
    gamma = torch.tensor([1.0, 1.0], dtype=torch.float16, requires_grad=True)
    beta = torch.tensor([0.0, 0.0], dtype=torch.float16, requires_grad=True)
    upstream = torch.tensor([[0.25, -0.75]], dtype=torch.float16)

    actual = layer_norm(x, gamma, beta, EPS)
    assert torch.isfinite(actual).all()

    (actual * upstream).sum().backward()
    actual_grads = (x.grad.detach().clone(), gamma.grad.detach().clone(), beta.grad.detach().clone())

    x_ref = x.detach().clone().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    expected = F.layer_norm(x_ref, (2,), gamma_ref, beta_ref, EPS)
    (expected * upstream).sum().backward()
    expected_grads = (x_ref.grad, gamma_ref.grad, beta_ref.grad)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), float("-inf")])
def test_rr02_functional_api_rejects_non_finite_epsilon(eps: float) -> None:
    """The functional API must reject every non-finite epsilon."""
    with pytest.raises(ValueError):
        layer_norm(torch.ones(2, dtype=torch.float32), torch.ones(2), torch.zeros(2), eps)


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), float("-inf")])
def test_rr03_module_api_rejects_non_finite_epsilon(eps: float) -> None:
    """The module constructor must apply the same epsilon contract."""
    with pytest.raises(ValueError):
        LayerNorm(2, eps=eps)


@pytest.mark.parametrize(
    ("x_dtype", "parameter_dtype"),
    [(torch.float16, torch.float32), (torch.float32, torch.float64)],
)
def test_rr04_dtype_policy_rejects_mixed_dtypes(
    x_dtype: torch.dtype, parameter_dtype: torch.dtype
) -> None:
    """Gate 013 freezes same-dtype x/gamma/beta semantics rather than mixed precision."""
    x = torch.ones((1, 2), dtype=x_dtype)
    gamma = torch.ones(2, dtype=parameter_dtype)
    beta = torch.zeros(2, dtype=parameter_dtype)
    with pytest.raises(ValueError, match="same dtype"):
        layer_norm(x, gamma, beta)


def _workflow_text() -> str:
    """Return the repository test workflow as text."""
    return WORKFLOW.read_text(encoding="utf-8")


def test_rr05_ci_covers_every_advertised_minimum_python_version() -> None:
    """Python 3.11 and 3.12 must both be exercised while >=3.11 is advertised."""
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    workflow = _workflow_text()
    assert 'requires-python = ">=3.11"' in pyproject
    assert "3.11" in workflow
    assert "3.12" in workflow


def test_rr06_workflow_declares_read_only_contents_permission() -> None:
    """CI must not inherit broader repository token permissions."""
    workflow = _workflow_text()
    assert "permissions:" in workflow
    assert "contents: read" in workflow


def test_rr07_checkout_does_not_persist_credentials() -> None:
    """Repository-controlled install/test code must not receive persisted checkout credentials."""
    workflow = _workflow_text()
    assert "persist-credentials: false" in workflow


def _documented_definition_coverage() -> tuple[int, int, float]:
    """Measure docstring coverage for functions and classes in changed Python surfaces."""
    documented = 0
    total = 0
    for root in (SRC, TESTS):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                total += 1
                if ast.get_docstring(node):
                    documented += 1
    coverage = 100.0 if total == 0 else documented * 100.0 / total
    return documented, total, coverage


def test_rr08_repository_docstring_coverage_is_at_least_80_percent() -> None:
    """Represent the PR quality warning as an executable repository quality threshold."""
    documented, total, coverage = _documented_definition_coverage()
    assert coverage >= 80.0, (
        f"docstring coverage {coverage:.2f}% ({documented}/{total}) is below the frozen 80% threshold"
    )
