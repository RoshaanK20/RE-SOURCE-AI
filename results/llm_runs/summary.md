# Experiment 2: LLM repair results

420 repair runs with free local models (qwen2.5-coder via Ollama) on the 21-case benchmark. Every distinct patch was reviewed by hand (`patch_audit.csv`): the verifier raised 1 false alarm(s) and missed 0 real change(s).

| Config | Mode | Runs | Silent method change | Verified reproduction | LLM calls |
|---|---|---|---|---|---|
| 7B, T=0 | Run-only | 76% | 24% | 52% | 2.10 |
| 7B, T=0 | Score ±10% | 76% | 14% | 52% | 2.48 |
| 7B, T=0 | RE-SOURCE verifier | 76% | 0% | 76% | 2.33 |
| 7B, T=0 | RE-SOURCE verifier + KB | 95% | 0% | 95% | 1.57 |
| 14B, T=0 | Run-only | 90% | 29% | 62% | 1.52 |
| 14B, T=0 | Score ±10% | 90% | 19% | 57% | 2.05 |
| 14B, T=0 | RE-SOURCE verifier | 86% | 0% | 86% | 2.00 |
| 14B, T=0 | RE-SOURCE verifier + KB | 100% | 0% | 100% | 1.24 |
| 7B, T=0.2 (3 seeds) | Run-only | 83% | 24% | 59% | 2.03 |
| 7B, T=0.2 (3 seeds) | Score ±10% | 81% | 14% | 59% | 2.35 |
| 7B, T=0.2 (3 seeds) | RE-SOURCE verifier | 81% | 0% | 81% | 2.33 |
| 7B, T=0.2 (3 seeds) | RE-SOURCE verifier + KB | 95% | 0% | 95% | 1.59 |
| All runs | Run-only | 83% | 25% | 58% | 1.94 |
| All runs | Score ±10% | 82% | 15% | 57% | 2.31 |
| All runs | RE-SOURCE verifier | 81% | 0% | 81% | 2.27 |
| All runs | RE-SOURCE verifier + KB | 96% | 0% | 96% | 1.51 |

Variation across the 3 seeds (7B, T=0.2), mean ± sd:

- Run-only: reproduction 59% ± 3%, silent change 24% ± 0%
- Score ±10%: reproduction 59% ± 3%, silent change 14% ± 0%
- RE-SOURCE verifier: reproduction 81% ± 5%, silent change 0% ± 0%
- RE-SOURCE verifier + KB: reproduction 95% ± 0%, silent change 0% ± 0%

Paired exact McNemar tests over all runs (pairs = same model, case, seed, temperature):

- verified reproduction: Run-only only 0, RE-SOURCE verifier + KB only 40, p = 1.8e-12
- verified reproduction: Score ±10% only 0, RE-SOURCE verifier + KB only 41, p = 9.1e-13
- verified reproduction: Run-only only 2, RE-SOURCE verifier only 26, p = 3e-06
- verified reproduction (effect of the KB): RE-SOURCE verifier only 0, RE-SOURCE verifier + KB only 16, p = 3.1e-05
- silent method change: Run-only only 26, RE-SOURCE verifier + KB only 0, p = 3e-08
- silent method change: Score ±10% only 16, RE-SOURCE verifier + KB only 0, p = 3.1e-05

Note: the verifier modes only accept repairs the verifier passes, so their 0% silent-change rate holds by construction. The informative comparisons are how often the baselines silently change the experiment, and how much the verifier's feedback and the knowledge base raise verified reproduction.
