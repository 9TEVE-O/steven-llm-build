# steven-llm-build

Building a small decoder-only language model from first principles, one verified capability at a time.

## Objective

The project aims to build a real, trainable language model while keeping important behaviours inspectable and evidence-bound. New capabilities are introduced behind explicit gates rather than assumed correct because the code resembles a conventional Transformer.

Engineering gate sequence:

> contract → support envelope → implementation → adversarial attack → static/security/quality checks → compatibility matrix → independent oracle → clean CI → external review → only then PASS

Generated code is not evidence of correctness. Git history, source code, executed tests and gate records are authoritative over conversational claims.

## Current repository state

This repository began as an empty shell on 8 September 2026. Historical experiments discussed outside the repository are not treated as repository evidence until their code/tests are recovered or reconstructed here and executed.

The current controlled gate is:

**Gate 013 — Layer Normalisation — REOPENED**

The initial candidate passed its frozen 25-test suite. Independent PR review subsequently identified untested contract and engineering failures, so Gate 013 was reopened and integration authority withdrawn.

The recovery fixture is frozen in `docs/gates/013_review_recovery_fixture.md`. The authoritative gate record is `docs/gates/013_layernorm.md`.

## Current scope

Gate 013 covers only:

- LayerNorm forward semantics
- explicit backward semantics for input, `gamma` and `beta`
- last-feature-axis behaviour
- degenerate and boundary inputs
- finite-difference gradient checks
- comparison with PyTorch as a reference oracle
- deliberately broken negative controls
- review-recovery attacks for FP16 accumulation, non-finite epsilon, dtype policy, compatibility, workflow security and repository quality

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

The project standard is:

> ChatGPT made a claim, built the strongest practical case against its own claim, and only published the claim that survived.

## Near-term roadmap

1. Recover Gate 013 under the frozen review-recovery fixture and engineering sequence.
2. Only after a new PASS, define the integration gate for LayerNorm placement in the decoder architecture.
3. Recover or reconstruct earlier Transformer components into repository-controlled evidence where needed.
4. Assemble a small decoder-only language model.
5. Train it on a bounded, documented corpus.
6. Add reproducible evaluation, checkpointing and generation experiments.
7. Scale only after the smaller system is understood.

## Author

Steven Lees

## Hosted-model integration

This project does not currently contain an OpenAI API integration or an active
hosted-model configuration. GPT-6 Astra is the requested target for a future
integration, but an API model identifier must be verified in the official OpenAI
documentation before it is configured.

An executable migration will require an existing API call site (including its
prompt and response contract) plus tests or evaluations that can validate the
model change. Do not infer an API model identifier from the display name.
