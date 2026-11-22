# RE-SOURCE AI | Scientific Integrity Gate

**Release candidate:** v0.4.0-rc3  
**Goal:** Assess the scientific integrity of a research-code repair *independently of its ML framework*.

## What is different from the legacy verifier?

The earlier verifier observes selected Scikit-learn calls and checks their runtime traces. It is still useful as a **controlled benchmark tool**, but it cannot establish scientific equivalence of arbitrary ML code.

`resource_ai/science.py` takes a different approach. It does not depend on Scikit-learn, TensorFlow, PyTorch, or JAX. It requires a **researcher-reviewed experiment contract** grounded in the original paper. It checks:

1. **Original source snapshot:** the complete reference-tree fingerprint matches the pre-recorded snapshot. The research commit is recorded in the contract; the reviewer must establish that the snapshot actually came from that commit.
2. **Protected scientific assets:** exact hashes of locked datasets, labels, configs, split definitions and evaluation files remain unchanged.
3. **Repair scope:** changed code must be exactly the pre-approved compatibility replacements. Unapproved modifications yield **UNVERIFIABLE**, not an automatic clean bill of health. Modified locked files yield **METHODOLOGY_VIOLATION**.
4. **Independent evaluation:** a reviewer-owned evaluator, outside the LLM-editable repository, runs in restricted Docker, or in an explicitly trusted dedicated local venv. Its file hash must match the contract. Docker requires a pinned image digest; local execution requires a reviewed `local_environment_sha256` manifest hash. No evaluator means **UNVERIFIABLE**.
5. **Measured result:** the independent metric is compared with the paper/reference result using a predeclared tolerance. Metric mismatch produces **RESULT_NOT_REPRODUCED**, not a silent success.

The gate cannot automatically understand the scientific intent of an arbitrary paper or prove semantic equivalence of all Python programs. A result `PASSED_SPECIFIED_SCIENTIFIC_CHECKS` means exactly that: **the specified checks passed**. It is not a claim that the full published paper has been reproduced.

## How to prepare a real research paper case

First inspect the paper and repository with **Scan a Link**, saving the actual commit. Use a **reviewed, isolated environment** to examine execution failures. Before seeing a repair's outcomes, record the experiment's original dataset, data partition, preprocessing, architecture, training process, evaluation procedure and outcome tolerance.

Prepare an independent Python evaluator, outside both the original and repaired source directories. It should load the original test data and independently compute its chosen metric from the repaired model, not copy the script's printed scores. It must print a JSON object to stdout, for example:

```json
{"metrics": {"accuracy": 0.842}}
```

This JSON is only an **interface example**, not a measured research result. An evaluator that blindly copies a candidate's claimed score is **not trustworthy**. The evaluator must be independently reviewed.

Generate a *draft* contract (the tool never automatically marks a contract as approved):

```powershell
python -m scripts.make_science_contract --source path/to/original_repo --evaluator path/to/trusted_evaluator.py --paper-title "Exact paper title" --repo-url https://github.com/owner/repo --commit 0123456789abcdef0123456789abcdef01234567 --framework pytorch --protect data/labels.csv --protect config/training.json --out contracts/paper.json
```

Review the generated `contracts/paper.json`, replace all `REVIEW_ME` fields with evidence from the paper, add preapproved compatibility replacements and a valid reference metric/tolerance. Only set `review_status` to `approved` after human review. Protect **all** relevant method files. If important configuration is inferred from Python rather than an external config, the unchanged-source check and exact approved edit rules are especially important.
For **every** dataset/split/preprocessing/model/training/evaluation section, also populate `evidence_files` with relevant paths from the original source tree. Dataset files must be explicitly hash-protected. Missing evidence files lead to `UNVERIFIABLE`.

Then run the external verifier (requires a locally available, reviewer-approved **SHA256-pinned** Docker image containing the project's dependencies):

```powershell
python -m resource_ai.science --contract contracts/paper.json --reference path/to/original_repo --candidate path/to/repaired_repo --evaluator path/to/trusted_evaluator.py --image "my-reviewed-image@sha256:<actual-64-character-sha256>" --out reports/scientific_result.json
```

A JSON and Markdown report are written. You can run the same checks in Streamlit under **Research Trial → Scientific Integrity Gate**.

### Optional trusted local evaluator (no Docker)

A local venv does NOT isolate untrusted research code. Only choose local evaluation for projects you have fully reviewed and trust. Create a separate research venv, review its dependencies, then calculate its manifest SHA256:

```powershell
py -3.13 -m venv research_env
.venv\Scripts\python.exe -m resource_ai.local_runtime --python research_env\Scripts\python.exe
```

Before evaluating a repair, copy the displayed SHA256 to the contract JSON as `local_environment_sha256`. This fingerprint covers the Python version, platform and installed package-version list, not individual library bytes. Changes to the environment invalidate the recorded hash.

```powershell
python -m resource_ai.science --backend local --trust-local --local-python research_env\Scripts\python.exe --contract contracts/paper.json --reference path/to/original_repo --candidate path/to/repaired_repo --evaluator path/to/trusted_eval.py --out reports/scientific_result.json
```

No matching approved lock means `UNVERIFIABLE`. The original source, dataset provenance, exact permitted patch, independent evaluator and actual measurement still need to be established. Neither local execution nor a matching score alone proves publication-level reproduction.

## What Streamlit actually supports

1. **Scan a Link / Research Queue**: download and inspect source; never automatically run an unknown repository.
2. **Research Trial**: with your explicit review, execute a Python entry point using restricted Docker or a dedicated, fully trusted local venv. If selected, Qwen can propose minimal code replacements on a copied project. A successful run alone is `EXECUTES_UNVERIFIED`.
3. **Research Trial → Scientific Integrity Gate**: examine the paper contract, snapshot, protected files, allowed patch scope and independently recomputed result, producing a separate scientific verdict.
4. **Controlled Recovery**: the older Scikit-learn-based demonstrations and real-data integrity regression tests. These are **not** claimed as recovered published papers.

## Validity and trust boundaries

- A repo URL alone does **not** establish the paper's dataset, labels, split, reference metric or expected architecture. A paper-specific contract is required.
- Merely identifying a compatibility issue is not evidence that a repository fails in practice.
- The original source commit is a declared identifier until its provenance is independently verified. The contract's tree digest prevents a later silent change, but cannot prove that the original snapshot came from the paper's repository.
- Docker restrictions reduce risk, but Docker is **not a perfect sandbox against malicious code**. Use a disposable machine/container environment, review images and repositories, and never supply personal credentials. Avoid arbitrary datasets or code from untrusted parties.
- The independent evaluator must not trust candidate-printed scores, and must keep labels/test data outside the agent's editable scope. Any unverifiable data provenance prevents a full scientific claim.
- Stochastic training may require multiple seeds and a predeclared statistical acceptance rule. The initial gate checks a single observed metric under a stated absolute tolerance and does **not** provide statistical proof.
- A successful Qwen repair must be visibly attributed to Qwen, with model name, prompt, response, patch and original failure; a scripted patch must not be counted as an LLM recovery.
- Live GitHub downloading, real Qwen and Docker are **not** independently exercised in this offline build. The automated science tests use toy source and a mocked evaluator. There are still no confirmed published-paper recoveries.

## Scientific verdicts

| Verdict | Interpretation |
|---|---|
| `PASSED_SPECIFIED_SCIENTIFIC_CHECKS` | Approved changes, protected files and external metric checks pass under the recorded evidence. Does not prove universal paper reproduction. |
| `METHODOLOGY_VIOLATION` | A protected dataset/config/reference file was modified. |
| `RESULT_NOT_REPRODUCED` | The independent measured metric does not match within the declared tolerance. |
| `UNVERIFIABLE` | Missing reference, unapproved edits, missing Docker/evaluator, unpinned image or another evidence gap. |

A separate execution trial may report `EXECUTES_UNVERIFIED` or `EXECUTION_FAILED`.
