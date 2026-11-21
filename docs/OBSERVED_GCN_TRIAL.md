# GCN repository: actual user-observed trial (21 November 2026)

**Paper/code candidate:** Kipf & Welling, Graph Convolutional Networks · https://github.com/tkipf/gcn

**Entry point observed:** `gcn/train.py` (not `setup.py` or root `train.py`). The user confirmed its location with PowerShell `Get-ChildItem -Recurse -Filter train.py`.

## What the user actually observed

| Evidence item | Observed result |
|---|---|
| GitHub intake | Source downloaded and scanned; source inspection only |
| Original research execution | **FAILED** |
| Python exception | `ModuleNotFoundError: No module named 'tensorflow'` |
| Qwen | A patch proposal was generated after the failure |
| Patch content | Malformed whole-file replacement containing `SEARCH` and `=======`, removing the original training loop/model/evaluation |
| Local patch execution | Local Python was configured not to auto-execute unreviewed generated edits |
| Scientific methodology | **UNVERIFIABLE** |
| Reproduced published paper | **No** |

**Interpretation:** The missing top-level TensorFlow package is a setup failure. It is *not evidence that the original paper is broken*. Replacing the training script with an import guard would not reproduce the model; that proposed edit is invalid.

**Change in the official release:** The research-trial runner now classifies a missing top-level framework as `ENVIRONMENT_BLOCKED` and does not request a Qwen source edit for it. It also rejects whole-file responses, malformed markers, syntax errors, and excessively broad modifications. These are **source-edit safety checks**, not proof of scientific equivalence.

**Evidence provenance:** This record is based on the user's copied Streamlit trial output, not a fresh network download or an independently rerun TensorFlow experiment. The original JSON report from the user's `runs/` directory should be archived privately with the original commit, environment versions and model tag. We do not fabricate missing commit or package-version information.

**Next scientifically valid step:** In an isolated environment, recreate the paper's documented runtime dependencies and datasets, establish baseline reference behavior, then attempt a genuine compatibility repair and independently evaluate it. Until that succeeds, the paper remains **NOT VERIFIED**.
