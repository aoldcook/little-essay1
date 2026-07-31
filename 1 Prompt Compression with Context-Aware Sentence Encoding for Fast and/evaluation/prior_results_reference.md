# Prior comparison results (recorded from earlier project run)

Recorded verbatim for reference. **These are NOT citable as current results** — see
the validity notes below before reusing any of these numbers.

Columns as supplied: Target (requested keep ratio), Actual CR, Ans Cov, Support Cov,
Hard Fact, QA Ans Rate, Utility/Token.

## HotpotQA

| Method | Target | Actual CR | Ans Cov | Support Cov | Hard Fact | QA Ans Rate | Utility/Token |
|---|---|---|---|---|---|---|---|
| Ours-Qwen | variable | 71.4 | 90.6 | 93.8 | 98.9 | 91.6 | 1.289 |
| Ours-Qwen | 0.20 | 22.7 | 69.9 | 63.6 | 84.2 | 62.0 | 3.309 |
| LLMLingua | 0.20 | 29.2 | 56.7 | 64.9 | 48.9 | 50.1 | 2.638 |
| LLMLingua2 | 0.20 | 17.6 | 61.4 | 36.4 | 54.6 | 8.0 | 2.900 |
| Ours-Qwen | 0.40 | 42.0 | 81.0 | 73.1 | 94.6 | 80.8 | 1.951 |
| LLMLingua | 0.40 | 49.9 | 68.3 | 76.9 | 63.6 | 68.5 | 1.689 |
| LLMLingua2 | 0.40 | 37.4 | 84.3 | 61.7 | 78.7 | 77.4 | 2.005 |
| Ours-Qwen | 0.60 | 60.3 | 88.2 | 89.1 | 97.7 | 89.6 | 1.491 |
| LLMLingua | 0.60 | 70.7 | 79.0 | 86.7 | 77.9 | 80.7 | 1.280 |
| LLMLingua2 | 0.60 | 57.3 | 91.4 | 80.8 | 90.7 | 93.7 | 1.505 |

## TriviaQA

| Method | Target | Actual CR | Ans Cov | Support Cov | Hard Fact | QA Ans Rate | Utility/Token |
|---|---|---|---|---|---|---|---|
| Ours-Qwen | variable | 19.5 | 82.4 | 86.5 | 84.7 | 83.1 | 5.251 |
| Ours-Qwen | 0.20 | 19.8 | 82.7 | 86.2 | 86.2 | 82.9 | 4.351 |
| LLMLingua | 0.20 | 22.3 | 76.6 | 75.8 | 59.7 | 77.4 | 3.950 |
| LLMLingua2 | 0.20 | 17.6 | 83.8 | 50.1 | 61.3 | 42.7 | 3.811 |
| Ours-Qwen | 0.40 | 36.8 | 87.7 | 92.7 | 93.0 | 88.4 | 2.445 |
| LLMLingua | 0.40 | 42.5 | 85.0 | 86.7 | 75.0 | 86.5 | 2.189 |
| LLMLingua2 | 0.40 | 37.4 | 91.4 | 76.1 | 75.4 | 91.4 | 2.217 |
| Ours-Qwen | 0.60 | 53.2 | 89.8 | 95.5 | 96.2 | 90.6 | 1.725 |
| LLMLingua | 0.60 | 62.6 | 89.4 | 93.1 | 85.6 | 90.7 | 1.528 |
| LLMLingua2 | 0.60 | 57.8 | 93.1 | 90.6 | 85.1 | 93.8 | 1.561 |

## Validity notes — read before reusing

1. **These are lexical coverage proxies, not downstream accuracy.** "Ans Cov",
   "Support Cov", "Hard Fact" and "QA Ans Rate" are token/entity overlap measures
   computed without a reader LLM. This is EVAL_VALIDITY_AUDIT.md finding C2, the
   defect the current downstream pipeline exists to remove. They cannot appear as
   headline results, and any improvement they show may not survive a real reader.

2. **The `variable` rows are at a different operating point and are not comparable
   to the fixed-ratio rows.** On HotpotQA, `Ours-Qwen variable` keeps 71.4% of
   tokens while the baselines it is tabled beside keep 17-29%. A method that keeps
   4x more text scoring higher coverage is expected, not evidence of quality.
   Comparisons must be made at matched Actual CR.

3. **At least one baseline row looks broken.** LLMLingua2 at target 0.20 on
   HotpotQA reports QA Ans Rate 8.0 while its Ans Cov is 61.4 — an implausible
   combination. On TriviaQA the same cell is 42.7. A baseline that collapses at
   one operating point usually indicates a configuration or truncation bug, not a
   weakness of the method. Reporting it as-is invites a reviewer to reject the
   whole comparison as an unfair baseline.

4. **Actual CR frequently misses Target.** Ours-Qwen at target 0.20 achieves 22.7
   on HotpotQA; LLMLingua at 0.60 achieves 70.7. Comparing methods at nominally
   equal targets but materially different achieved ratios confounds the
   compression axis with the quality axis.

5. **Split provenance unknown.** These predate the group-disjoint re-split. The
   split in use at the time leaked 13.1% of test contexts into training.

6. Dataset scope differs from the current pool: HotpotQA and TriviaQA here, versus
   the mixed CQR/HotpotQA/LongBench/WikiText pool used now.

**Usable as:** a sanity reference for the direction and rough magnitude of prior
results, and a record of which baselines were run.
**Not usable as:** results in the paper, or a baseline to claim improvement over.
