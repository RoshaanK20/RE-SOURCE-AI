# RE:SOURCE AI | Teacher demonstration (first official pilot)

## Show the actual evidence, not predicted results

**One-minute explanation:**

> RE:SOURCE AI investigates why older machine-learning experiments fail when run in newer software environments. It uses a local Qwen model to propose code repairs, but code that runs is not automatically a scientifically valid repair. We therefore record the patch, execution result, and methodology checks separately. Our first pilot has measured tests on real datasets, and we have an observed failed trial on an actual published GCN repository. The GCN trial exposed a missing TensorFlow dependency and an invalid AI patch, which our updated release now prevents. Independent reproduction of the GCN paper is still pending.

## Demonstration sequence

1. **Reports & About:** show the measured Breast Cancer and Wine integrity tests. Explain that real datasets are used with intentionally created integrity violations; don't present them as recovered research papers.
2. **Controlled Recovery:** show original vs broken vs repaired Diabetes experiment; label the successful edit **SCRIPTED**, not Qwen-generated. Show the measured R² and saved JSON.
3. **Qwen diagnostic (optional live):** `python -m scripts.check_llm --model qwen2.5-coder:7b --smoke`; then attempt a controlled Qwen case and show the raw output/diff. Record failure honestly if the model doesn't repair it.
4. **Research Queue:** inspect GCN and other source links. This is **real repository source inspection**, not training or replication.
5. **Scientific Integrity Gate:** explain that it needs a paper-grounded contract, original snapshot, approved minimal changes, and separate trusted evaluator. Do not fill in placeholders or issue a passing verdict without evidence.

## Three evidence categories

| Type | What we know | What we must NOT say |
|---|---|---|
| Controlled real-data integrity experiments | Test reports demonstrate handling of specified correct/incorrect changes | Universal scientific validation for any framework |
| Controlled Diabetes compatibility demo | Scripted compatibility correction works on a real dataset | Qwen repaired a published paper |
| Observed GCN user trial | Original run lacked TensorFlow; malformed Qwen edit was proposed | Successful autonomous GCN recovery or reproduced research |

## Likely questions and defensible answers

**Why not use score matching alone?** Because fabricated or improperly evaluated predictions can produce plausible metrics. Our controlled tests exercise specific such violations.

**Is Qwen autonomous?** The bounded Qwen repair loop is implemented and its mocked integration has regression tests. A successful live local Qwen trial must be separately demonstrated; it cannot be inferred from a simulated response.

**Does it handle PyTorch and TensorFlow?** It can inspect Python files and compare paper-specific methodology contracts across frameworks. It does not automatically verify arbitrary PyTorch/TensorFlow experiments or migrate old GCN models.

**Is a failed trial useful?** Yes. The missing-package failure let us distinguish environment repair from source-code repair, and the malformed edit exposed a safety problem. Both are explicitly recorded as failures, not successes.

**Have you already reproduced a published research paper?** No. The real published GCN repository has been inspected and a failed local execution was observed, but independently verified reproduction is still an open experiment.
