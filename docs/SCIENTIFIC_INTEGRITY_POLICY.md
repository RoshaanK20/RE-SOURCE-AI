# Scientific integrity policy (experimental)

RE-SOURCE AI distinguishes:

1. **Execution**: the script finished without an exception.
2. **Observed methodological agreement**: currently supported recorder events agree with a reference trace.
3. **Result agreement**: the recorded metrics fall within a stated comparison rule.
4. **Independent scientific reproduction**: requires trusted historical datasets, reference protocol, and evaluation outside the candidate's edit scope. **This is not yet established by the current recorder.**

The current verifier checks recorded estimator fits and parameters, input/label fingerprints, training pairings, data splits, test-set predictions, selected metric inputs and test feature fingerprints. It flags unsupported claims. It can still be bypassed by code that tampers with instrumentation or fabricates whole model events: recorder traces are observations from the same experiment process, not independently certified data.

## What can be changed in a repair?

- Permitted: API spelling or dependency compatibility updates that are demonstrably equivalent to the original method; harmless parallelism settings when outputs and execution are appropriately checked.
- Not automatically permitted: changing datasets or splits, shuffling sample-label associations, changing model class or meaningful hyperparameters, changing preprocessing semantics or evaluation, or using test samples for training.
- If original methodology is unavailable, the result should be **unverifiable** rather than a claim of reproduction.

## Trust boundaries

The model and repair process must not control the reference, test labels, rules or independent result evaluator. The current subprocess runner imposes time limits but is **not a security sandbox**. Do not run unreviewed third-party repositories inside a personal Colab session or on a host containing credentials.

## Validation distinction

- **Controlled tests on real datasets**: actual Breast Cancer Wisconsin and Wine observations, with deliberately modified scripts. Tests recorder behavior on real data, not generalization to historical research.
- **Genuine repository recovery**: requires independently chosen published code, source commit, original environment, reference methods/data, real failure reproduction, license review, independent scoring. No such cases are included in the current release.

## Provisional verdict terminology

`REPRODUCED` in the existing API means "passed the supported runtime-trace and result checks," **not** "independently certified to reproduce the original paper." A future stable API should use separate `RAN`, `METHOD_OBSERVED`, `RESULT_MATCH`, and `INDEPENDENTLY_VERIFIED` states.

## v0.4.0-rc2 update: framework-independent science gate

`resource_ai.science` is a separate paper-specific verifier independent of ML framework. It checks a pre-reviewed contract and source/data/evaluator evidence. The older `verify.py` Scikit-learn trace checker remains in the package only for existing controlled regression cases. The new gate **never** interprets a run-only success as scientific reproduction and uses `PASSED_SPECIFIED_SCIENTIFIC_CHECKS` rather than `REPRODUCED` for its strongest verdict. The declared paper commit must be independently linked to the original source snapshot; the tree hash alone does not establish repository provenance. See `docs/FRAMEWORK_INDEPENDENT_SCIENCE.md`.
