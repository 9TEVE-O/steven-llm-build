# Engineering Instructions

## Governing Rule

Do not convert implementation into evidence. Code being present, compiling or executing does not establish that the intended behaviour is correct.

## Frozen Engineering Gate Sequence

For each controlled capability, use this sequence:

> contract → support envelope → implementation → adversarial attack → static/security/quality checks → compatibility matrix → independent oracle → clean CI → external review → only then PASS

No earlier stage independently authorises PASS.

1. Freeze the smallest sufficient behavioural contract.
2. Freeze the support envelope being claimed: dtypes, devices, runtime versions, shapes, magnitude ranges and other material boundaries.
3. Implement only the capability required by the open gate.
4. Construct adversarial attacks against the claim, including boundary, degenerate and deliberately defective controls where useful.
5. Run static, security and repository-quality checks.
6. Exercise every runtime configuration the repository advertises as supported.
7. Compare against an independent oracle where one exists.
8. Obtain clean CI evidence for the frozen attacks and checks.
9. Perform a pre-review against the classes of criticism expected from independent reviewers such as Copilot, Codex and CodeRabbit.
10. Submit to external review.
11. Assign PASS only if the claim survives the applicable evidence above.
12. Integrate only after standalone requirements pass, then rerun affected regression tests and record the result.

A green mathematical test suite is evidence, not gate authority. A valid defect found during review keeps the gate FAIL or UNRESOLVED until the defect is represented by a reproducible attack, corrected and rerun through the applicable sequence.

## Adversarial Pre-Review

Before a public PR is described as PASS, perform an internal review that represents at least these criticism classes:

- contract/implementation mismatch
- support-envelope gaps, including dtype, magnitude and non-finite values
- forward and backward numerical correctness
- boundary and degenerate behaviour
- compatibility between advertised and exercised runtimes
- workflow least privilege and credential handling
- static analysis and repository-quality findings
- missing or weak independent-oracle evidence
- stale gate status, unsupported completion claims or integration authority

This is not a bot-specific optimisation step. The purpose is to represent the same classes of engineering criticism before publication.

## Scope Control

Work only on the current open gate. Do not introduce unrelated architectural improvements while implementing a gate.

## Claims

Use evidence-strength-appropriate language. Prefer claims such as “passes the tested cases” or “matches the reference within the frozen tolerance”. Preserve failures and uncertainty.

The project standard is:

> ChatGPT made a claim, built the strongest practical case against its own claim, and only published the claim that survived.

## AI-Assisted Development

AI-generated code receives no special evidentiary status. Inspect it, execute it and test the properties being claimed. Git history, source code, executed tests and gate records are authoritative over conversational claims.

## Completion

Completing a primitive does not establish that an integrated Transformer works. Completing a Transformer does not establish successful training. Successful training does not establish general capability.
