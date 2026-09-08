# steven-llm-build

Building a small decoder-only language model from first principles, one verified capability at a time.

## Objective

The project aims to build a real, trainable language model while keeping important behaviours inspectable and evidence-bound. New capabilities are introduced behind explicit gates rather than assumed correct because the code resembles a conventional Transformer.

Development sequence:

> specify → implement → attack → verify → integrate → regress

Generated code is not evidence of correctness. Git history, source code, executed tests and gate records are authoritative over conversational claims.

## Current repository state

This repository began as an empty shell on 8 September 2026. Historical experiments discussed outside the repository are not treated as repository evidence until their code/tests are recovered or reconstructed here and executed.

The current controlled gate is:

**Gate 013 — Layer Normalisation**

LayerNorm is being established as a standalone primitive before any Transformer integration. See `docs/gates/013_layernorm.md` for the frozen contract, attacks and acceptance criterion.

## Current scope

Gate 013 covers only:

- LayerNorm forward semantics
- explicit backward semantics for input, `gamma` and `beta`
- last-feature-axis behaviour
- degenerate and boundary inputs
- finite-difference gradient checks
- comparison with PyTorch as a reference oracle
- deliberately broken negative controls

It does **not** yet cover LayerNorm placement inside a Transformer, causal-regression testing, training behaviour or model-quality claims.

## Repository layout

```text
src/steven_llm/       implementation
tests/                executable tests
docs/gates/           gate contracts and evidence records
.github/workflows/    continuous integration
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
pytest
```

## Engineering rules

Read `CONTRIBUTING.md` before modifying code.

The governing rule is simple: implementation does not establish correctness. Claims are limited to what executed evidence supports.

## Near-term roadmap

1. Complete Gate 013 standalone LayerNorm.
2. Only after PASS, define the integration gate for LayerNorm placement in the decoder architecture.
3. Recover or reconstruct earlier Transformer components into repository-controlled evidence where needed.
4. Assemble a small decoder-only language model.
5. Train it on a bounded, documented corpus.
6. Add reproducible evaluation, checkpointing and generation experiments.
7. Scale only after the smaller system is understood.

## Author

Steven Lees
