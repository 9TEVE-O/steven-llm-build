# Engineering Instructions

## Governing Rule

Do not convert implementation into evidence. Code being present, compiling or executing does not establish that the intended behaviour is correct.

## Development Sequence

For each new capability:

1. State the capability being introduced.
2. Freeze the smallest sufficient behavioural contract.
3. Implement the capability independently where practical.
4. Construct positive tests.
5. Construct boundary and degenerate tests.
6. Construct independent reference checks.
7. Introduce deliberately defective implementations where useful.
8. Confirm that the tests reject those defects.
9. Assign PASS, FAIL or UNRESOLVED from executed evidence.
10. Integrate only after standalone requirements pass.
11. Rerun affected regression tests.
12. Record the result.

## Scope Control

Work only on the current open gate. Do not introduce unrelated architectural improvements while implementing a gate.

## Claims

Use evidence-strength-appropriate language. Prefer claims such as “passes the tested cases” or “matches the reference within the frozen tolerance”. Preserve failures and uncertainty.

## AI-Assisted Development

AI-generated code receives no special evidentiary status. Inspect it, execute it and test the properties being claimed. Git history, source code, executed tests and gate records are authoritative over conversational claims.

## Completion

Completing a primitive does not establish that an integrated Transformer works. Completing a Transformer does not establish successful training. Successful training does not establish general capability.
