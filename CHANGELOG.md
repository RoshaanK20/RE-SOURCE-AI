# Changelog

## v0.4.0: paper release (ICCRT-2026)

- Added the evidence behind the paper: logs of all 420 repair runs, the manual patch audit, the 51-fix verifier experiment and the real-data check, under `results/`.
- Added `experiments/paper_results.py` (rebuilds the paper's tables and Fig. 2 from the logs) and `experiments/reverify_runs.py` (re-judges all logged patches with the current verifier).
- Added an MIT `LICENSE`, a `.gitattributes` file for consistent line endings, and author, citation and paper-results sections in the README.
- Renamed the project to RE-SOURCE AI throughout and replaced the earlier demonstration notes with `docs/DEMO_GUIDE.md`.

## v0.4.0-rc3 — Docker optional for trusted research trials

- Added a dedicated local-virtual-environment execution backend to the Streamlit Research Trial. Docker remains available and the default backend in the Python API.
- Added explicit user review and trust confirmations; a virtual environment is explicitly **not** represented as a security boundary.
- Preserved Qwen patching and execution records across both backends; execution remains `EXECUTES_UNVERIFIED` without independent science checks.
- Added local independent-evaluator support requiring a paper-specific contract, pinned evaluator hash and approved `local_environment_sha256` (version-manifest fingerprint). Missing locks cannot pass.
- Added actual temporary-venv integration tests; no claim of live Qwen, live GitHub execution or real published-paper recovery.

## 0.3.0-alpha | Research Recovery Console

- Added a Streamlit application (`app.py`) with a single-link inspector, Markdown/CSV research queue, live sequential scan progress, charts based on real inspection counts, report downloads, and local case history.
- Reused the existing static GitHub intake, compatibility knowledge base, methodology verifier, and Qwen repair engine.
- Added optional, explicitly invoked Qwen repair for the bundled trusted benchmark cases, and a deterministic scripted demonstration on the real Diabetes dataset.
- Added headless dashboard tests, including GitHub-only queue validation and exporting reports without source code.
- Kept unknown public GitHub source **static inspection only**; never auto-execute unknown repositories.
- The new interface is an alpha release. Real historical-repository reproduction and independent scientific certification remain unverified.

---

## 0.2.2-alpha (experimental development snapshot)

- Continued the existing modular repair agent and 21-case constructed benchmark; no redesign of the working LLM interface.
- Extended the Scikit-learn recorder to retain prediction-output fingerprints, metric-input fingerprints, and training feature/label pair fingerprints.
- Added verifier checks for fabricated metric inputs, changed evaluation features, and changed label associations.
- Excluded incidental worker-thread internal tree predictions from top-level comparison (so legitimate `n_jobs` changes are not falsely rejected).
- Fixed score-mode acceptance when no comparable metric is available.
- Added real-dataset controlled integrity checks using Breast Cancer Wisconsin and Wine, plus an independent JSON report.
- Added regression tests, tested dependency record and GitHub documentation.
- Kept explicit limits on verifying arbitrary Python, real historical research reproduction, and PyTorch.

## Previous development (historical milestones)

- Learning Tasks 1-3: an Iris Random Forest reference, a deliberately broken script, and a rule-based diagnosis notebook.
- Earlier modular prototype: reference/recorder, subprocess execution, scientific verifier, patching and repair-agent loop.
- Later alpha baseline: a local-model repair interface and 21 deliberately constructed benchmark cases.

> Version tags describe repository development, not research-paper results. Earlier uploaded notebooks are retained in development backups; they are excluded from the lean public release ZIP and are not independent scientific validations.

## v0.2.2-alpha: official pilot intake extension

- Added URL/file intake with research provenance report and static code inventory.
- Added Colab supervisor demonstration: paste code link and generate `CASE_REPORT.md`/`case.json`.
- Added deterministic controlled repair demonstration using real Scikit-learn Diabetes data.
- Added candidate-paper queue and source safety tests.
- Kept downloaded public repository execution disabled pending isolated runner and trusted evaluation.
- This is an enhancement to the same alpha, not a claim of publication-ready V3.

### v0.2.2-alpha | Batch research-source intake (incremental pilot)
- Added `resource_ai/batch.py` for automatic sequential processing of a CSV, Markdown, text or JSON list of research-code links.
- Added editable `corpus/PAPERS_TO_SCAN.md` containing five *unverified* paper/repository candidates.
- Added `batch-inspect` command and a dedicated Colab batch-runner notebook.
- Added per-case error isolation, consolidated Markdown/CSV/JSON reports, and offline/mock-backed regression tests.
- Maintained the scientific boundary: arbitrary downloaded GitHub code is inspected but never executed.
- No new scientific reproducibility claims or confirmed real-repository recoveries.

## 0.4.0-rc1 | Submission candidate (November 2026)

- Added explicit local Ollama diagnostics: reachable server, exact installed Qwen tag, optional real completion check, and differentiated connection/model/response errors.
- Added a command-line diagnostic (`python -m scripts.check_llm --smoke`). A simulated Ollama HTTP server is used in local regression tests; it is never presented as a live Qwen experiment.
- Added live activity feedback and downloadable model-run evidence (source SHA-256, model, environment, agent attempts and failures).
- Added a real-dataset validation table and chart backed by measured JSON; no fictional results are plotted.
- Added a one-command evidence runner (`python -m scripts.verify_release`).
- Added first-public-release supervisor demo, published-code screening status and updated scientific limitations.
- External repository source execution remains disabled; no independently verified published-paper reproduction is claimed.

## v0.4.0-rc2 — Framework-independent Scientific Integrity Gate

- Added `resource_ai/science.py`, a paper-specific integrity verifier that is **not tied to any ML framework**. Requires a researcher-approved methodology contract, locked original-source snapshot, protected scientific files, explicit file-role evidence, exact approved edits and a separate pinned evaluator.
- Added `resource_ai/framework_trial.py`, the reviewed-source Docker trial runner for Python ML research entry points. No execution on the host; successful execution is explicitly `EXECUTES_UNVERIFIED` until scientific evaluation passes. Docker is required for a real external trial.
- Added **Research Trial** to the Streamlit app, with separate execution and methodology verdicts, evidence tables, measured metric comparison charts and downloadable JSON/Markdown reports.
- Added `scripts/make_science_contract.py` to create *draft*, not automatically approved, paper contracts.
- Added cross-framework regression tests for PyTorch-, TensorFlow-, JAX-, and NumPy-style source, unapproved edits, protected-data tampering, evaluator evidence and unsafe entry points.
- Kept legacy Scikit-learn-based runtime tests as regression evidence, not a universal scientific-verification claim.
- **Still pending:** live GitHub retrieval, live Docker trials, actual Qwen repair on new repositories, independent published-paper recovery, and statistical multi-seed evaluation. No previously generated result has been relabeled as a published-paper reproduction.

## Official evidence-first submission

- Kept existing source scanner, paper queue, bounded Qwen agent, Streamlit dashboard and original benchmarks.
- Fixed cross-framework Research Trial to treat **missing top-level ML framework dependencies** as an environment blocker, not a source-code bug.
- Rejected research-trial whole-file LLM replacements, malformed patch markers, invalid syntax and disproportionately large edits.
- Allowed the original baseline to be recorded when Ollama is offline, instead of blocking every trial.
- Added clear Streamlit validation of methodology contract/evaluator paths and explicit failure statuses.
- Added supervisor demonstration notes and a user-observed GCN failure report; no claimed published-paper reproduction.
- No new model families, frameworks or untested paper recoveries added to the official scientific claims.

## Review fixes (21 November 2026)

- Fixed the verifier marking correct repairs as `NO_EVIDENCE` when they contain unsupervised steps (scalers, imputers, clustering). Feature/label pairing evidence is now only required for supervised fits. Before the fix, 18 of 30 correct benchmark fixes were accepted; after it, 29 of 30.
- The recorder now also observes predictions made through `GridSearchCV` and other scikit-learn methods wrapped with `available_if`.
- Added a regression test for both cases in `tests/test_resource_ai.py`.
- Restored 41 benchmark files that were missing from the shared copy (regenerated with `benchmark/build_cases.py`; identical to the recorded SHA256 manifest).
