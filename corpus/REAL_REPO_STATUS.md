# Published research repository screening status

The candidate links are maintained in `PAPERS_TO_SCAN.md`. **None has been confirmed as a reproduced failure or a scientifically recovered published result.**

| Case | Paper-code candidate | Paper methodology independently checked? | Original failure reproduced? | LLM repair verified? |
|---|---|---|---|---|
| P01 | tkipf/gcn | No | No | No |
| P02 | tkipf/gae | No | No | No |
| P03 | yoonkim/CNN_sentence | No | No | No |
| P04 | facebookresearch/InferSent | No | No | No |
| P05 | hexiangnan/neural_collaborative_filtering | No | No | No |

The included batch queue can read and inspect these repositories when GitHub access is available. It cannot infer their original experimental requirements from a GitHub URL alone. Many use unsupported legacy deep-learning frameworks; a Scikit-learn verifier verdict must not be extended to them.

Before accepting any real-world research case: pin the paper and GitHub commit, obtain original dataset and permitted setup, establish original execution conditions, observe and save the actual failure, run repaired code in a disposable restricted environment, and compare against a trusted independent reference. Include failed cases as well as successes.
