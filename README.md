# RE:SOURCE AI | Official Research Pilot

Experimental research software · **Submission date: 23 November 2026**

RE:SOURCE AI investigates compatibility failures in older Python ML research code, uses locally running **Qwen via Ollama** to propose minimal code patches, and checks whether an attempted repair has sufficient evidence to preserve the original scientific methodology. The app is a Streamlit dashboard designed to run in VS Code on Windows. **Docker is optional.**

> **Research claim, carefully scoped:** The project's scientific-integrity checks can distinguish selected valid and invalid modifications on measured *controlled* experiments using genuine datasets. **No independently verified recovery of a published research paper is included in this release.**

## 1. Start here (Windows / VS Code)

Extract the ZIP, then open the **`RE_SOURCE_AI`** folder in VS Code. Run the following from its terminal (Python 3.12 or 3.13 recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard normally opens at `http://localhost:8501`. Run Streamlit with the **app** virtual environment, not the separate `research_env`.

To check real Qwen connectivity (with Ollama installed and running):

```powershell
ollama list
ollama run qwen2.5-coder:7b "Reply READY"
.\.venv\Scripts\python.exe -m scripts.check_llm --model qwen2.5-coder:7b --smoke
```

If Qwen is unavailable, **repository scanning, test reports, and the scripted demonstration still work**. Qwen generation and a real autonomous repair are separate checks and cannot be claimed without actual logs.

## 2. Five screens, one workflow

| Streamlit screen | Action | Evidence level |
|---|---|---|
| **Scan a Link** | Read public GitHub source and inventory files | Inspection only |
| **Research Queue** | Read all links from `corpus/PAPERS_TO_SCAN.md` automatically | Batch inspection only |
| **Research Trial** | Run *reviewed* code in Docker or a trusted local research virtual environment | Actual execution, **not** scientific reproduction |
| **Controlled Recovery** | Show recorded real-dataset tests; optionally run genuine Qwen on a bundled compatibility case | Controlled experimental evidence |
| **Reports & About** | Display measured charts and downloadable JSON reports | Recorded results only |

### Source-execution safety

Unknown GitHub projects **must not run directly on your Windows computer**. A Python virtual environment separates packages but does **not** isolate files, credentials or network. Use a suitably restricted/disposable execution environment for unfamiliar research code. The Docker backend is optional, but local mode is allowed **only for code and dependencies you've reviewed and trust**. In local mode, Qwen patches are **proposals** and are **never executed automatically**.

## 3. What the first official tests prove

Use the release results under `results/` or regenerate them:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_release --full-suite
```

This runs Python regression tests, scientific-integrity challenges on the **Breast Cancer Wisconsin and Wine datasets**, and a scripted Random Forest compatibility repair on the **Diabetes dataset**. Outputs come from subprocess exit codes and actual measurements and are written to `results/release_evidence.json`, `results/official_validation_run.json`, and `results/official_controlled_test.json`.

These experiments use **real datasets but deliberately constructed errors**. The scripted Diabetes correction is **not** Qwen-generated. Source-trace checks operate on selected Scikit-learn tasks and do **not** generalize automatically to any framework.

## 4. Real research repository: observed GCN trial

The public research-code candidate **Graph Convolutional Networks** is linked at https://github.com/tkipf/gcn. A user-observed local trial found `gcn/train.py` and produced `ModuleNotFoundError: No module named 'tensorflow'`. This demonstrates an **environment dependency failure**, not a confirmed bug in the original paper. Qwen later proposed a malformed, destructive replacement containing patch markers; that proposal **must be rejected**, and the candidate was not shown to reproduce the paper.

The release now blocks both failure modes: missing top-level ML packages are reported as **ENVIRONMENT_BLOCKED** (no code rewrite is requested), and malformed/large whole-file replacements are rejected in the research-trial repair path. See `docs/OBSERVED_GCN_TRIAL.md` for the exact evidence and limits. This recorded observation was supplied from a Windows trial; we have **not** independently reproduced GCN here.

## 5. Framework-neutral scientific integrity

`resource_ai/science.py` verifies a **researcher-approved paper methodology contract**, protected dataset/configuration hashes, explicitly approved source edits, and measurements from a **separate, pinned evaluator**. It can inspect Python source from various frameworks, but it does **not** automatically infer a published paper's methodology or establish scientific equivalence from a GitHub URL alone.

The verdict `PASSED_SPECIFIED_SCIENTIFIC_CHECKS` means **only the supplied evidence contract checks passed**. When the paper's reference methodology, original data, approved changes, or independent evaluation are missing, the outcome must be `UNVERIFIABLE` rather than `REPRODUCED`. For setup details, see `docs/FRAMEWORK_INDEPENDENT_SCIENCE.md`. **Do not use the Scientific Integrity Gate before the original execution and paper methodology contract are prepared.**

### Known limits

- A successful script execution is not proof of scientific reproduction.
- Live Ollama, GitHub availability, and the Streamlit browser appearance must be checked on your Windows computer; they cannot be verified in the offline release environment.
- The full scientific verifier needs paper-specific evidence and an evaluator that is independent of the repair agent.
- GPU deep-learning runs, old TensorFlow compatibility, and independently confirmed published-paper recovery are future evaluation tasks, **not results** of this pilot.
- **Do not claim universal PyTorch, TensorFlow, or JAX scientific validation** based on mocked or source-contract unit tests.

**Research question:** Can an autonomous compatibility-repair agent distinguish a valid restoration of the original experiment from a repair that merely makes code execute?


## Authors

| Name | Role | Affiliation |
|---|---|---|
| Roshaan Ahmed Khan | Lead developer / Researcher | Department Of Artificial Intelligence, Sindh Madressatul Islam University, Karachi, Pakistan |
| Safa | Developer / Researcher | Department Of Artificial Intelligence, Sindh Madressatul Islam University, Karachi, Pakistan |

**Contact:** roshaanahmed2005@gmail.com
**Contact:** safakhan.tech@gmail.com

## Citation

If you use this work, please cite:

> Roshaan Ahmed Khan, Safa. *RE:SOURCE AI: An Agent for Recovering and Verifying Machine Learning Experiments.* Sindh Madressatul Islam University, 2026.
