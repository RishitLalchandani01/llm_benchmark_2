# Evaluation summary

## Abstract placeholders (primary ladder, all items)

- [X] accuracy gain qwen2.5-0.5b → qwen2.5-3b: **36.6 points**
- [Y] accuracy gain qwen2.5-3b → qwen2.5-14b: **27.8 points**
- [Z] inference compute per response, qwen2.5-14b / qwen2.5-0.5b: **36.2×**

## Per-model metrics (all items, greedy run; variance from sampled runs)

| model_id | ladder | params_b | accuracy | ci | format_failure_rate | variance | mean_output_tokens | ter | U_points_per_tflop | semantic_divergence | bertscore_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3-270m | comparator | 0.27 | 2.2 | [1.1, 3.6] | 91.0 | 0.006 | 91 | 24.6 | 16.4 | 0.566 | 0.434 |
| gemma3-1b | comparator | 1 | 29.1 | [24.3, 33.9] | 22.3 | 0.0441 | 266 | 109 | 33.9 | 0.323 | 0.884 |
| gemma3-4b | comparator | 1 | 50.8 | [45.7, 55.9] | 19.7 | 0.0176 | 205 | 247 | 68.9 | 0.119 | 0.93 |
| gemma3-12b | comparator | 1 | 55.4 | [50.0, 60.7] | 10.3 | 0.0256 | 196 | 283 | 77 | 0.122 | 0.933 |
| llama3.2-1b | comparator | 1.24 | 15.5 | [11.6, 19.6] | 56.3 | 0.0627 | 165 | 94 | 18.4 | 0.343 | 0.747 |
| qwen2.5-coder-1.5b | comparator | 1.54 | 20.2 | [16.0, 24.5] | 47.7 | 0.0915 | 248 | 81.5 | 16 | 0.216 | 0.891 |
| gemma2-2b | comparator | 2.61 | 30.4 | [25.6, 35.4] | 20.3 | 0.0411 | 133 | 229 | 19.7 | 0.184 | 0.912 |
| qwen2.5-coder-3b | comparator | 3.09 | 34.4 | [29.2, 39.6] | 25.7 | 0.0846 | 197 | 175 | 15.5 | 0.205 | 0.918 |
| llama3.2-3b | comparator | 3.21 | 40.8 | [35.4, 46.1] | 27.7 | 0.0348 | 164 | 248 | 18.8 | 0.155 | 0.927 |
| phi3-mini | comparator | 3.82 | 44.0 | [38.9, 49.0] | 21.0 | 0.0906 | 209 | 211 | 15.1 | 0.167 | 0.911 |
| phi4-mini | comparator | 3.84 | 42.3 | [37.2, 47.3] | 21.0 | 0.111 | 120 | 353 | 20.2 | 0.187 | 0.914 |
| llama2-7b | comparator | 6.74 | 22.3 | [18.2, 26.7] | 17.3 | 0.054 | 181 | 123 | 4.63 | 0.286 | 0.9 |
| llama2-13b | comparator | 6.74 | 23.5 | [19.1, 28.0] | 17.7 | 0.0577 | 196 | 120 | 4.68 | 0.295 | 0.894 |
| codellama-7b | comparator | 6.74 | 22.7 | [18.5, 27.0] | 23.7 | 0.0456 | 173 | 132 | 4.82 | 0.257 | 0.895 |
| mistral-7b | comparator | 7.25 | 28.0 | [23.6, 32.5] | 15.3 | 0.0766 | 153 | 182 | 6.06 | 0.25 | 0.896 |
| qwen2.5-coder-7b | comparator | 7.61 | 47.1 | [41.9, 52.2] | 21.3 | 0.0627 | 193 | 244 | 8.68 | 0.141 | 0.924 |
| llama3-8b | comparator | 8.03 | 35.3 | [30.3, 40.4] | 22.3 | 0.039 | 121 | 292 | 7.72 | 0.15 | 0.914 |
| gemma2-9b | comparator | 9.24 | 43.2 | [38.0, 48.3] | 18.0 | 0.0464 | 119 | 364 | 8.31 | 0.119 | 0.928 |
| mistral-nemo-12b | comparator | 12.2 | 58.9 | [53.8, 64.0] | 2.7 | 0.101 | 117 | 503 | 8.88 | 0.159 | 0.919 |
| codellama-13b | comparator | 13 | 25.1 | [20.7, 29.7] | 23.3 | 0.0467 | 161 | 156 | 2.86 | 0.228 | 0.898 |
| phi3-medium | comparator | 14 | 48.5 | [43.4, 53.5] | 18.7 | 0.069 | 199 | 244 | 4.66 | 0.138 | 0.92 |
| phi4-14b | comparator | 14.7 | 69.1 | [64.5, 73.6] | 7.3 | 0.0518 | 313 | 221 | 4.91 | 0.109 | 0.927 |
| qwen2.5-coder-14b | comparator | 14.7 | 58.9 | [53.9, 63.9] | 18.0 | 0.0578 | 264 | 223 | 4.69 | 0.11 | 0.927 |
| mistral3-nemo-24b | comparator | 23.6 | 72.9 | [68.6, 77.1] | 3.7 | 0.0444 | 274 | 266 | 3.59 | 0.104 | 0.932 |
| qwen2.5-0.5b | primary | 0.49 | 9.1 | [6.2, 12.2] | 61.7 | 0.0506 | 160 | 56.9 | 28.7 | 0.449 | 0.627 |
| qwen2.5-1.5b | primary | 1.54 | 13.8 | [10.1, 17.7] | 70.0 | 0.065 | 157 | 87.7 | 14 | 0.397 | 0.642 |
| qwen2.5-3b | primary | 3.09 | 45.7 | [40.3, 51.1] | 21.7 | 0.0766 | 169 | 271 | 22.3 | 0.184 | 0.863 |
| qwen2.5-7b | primary | 7.61 | 54.4 | [49.1, 59.8] | 12.3 | 0.0597 | 162 | 336 | 11 | 0.111 | 0.929 |
| qwen2.5-14b | primary | 14.7 | 73.5 | [69.3, 77.6] | 1.3 | 0.0322 | 227 | 324 | 6.41 | 0.116 | 0.922 |
| llama3.1-8b | secondary | 8.03 | 49.2 | [44.0, 54.5] | 19.3 | 0.0599 | 185 | 266 | 8.78 | 0.123 | 0.922 |

## Marginal utility per doubling (Eq. 6) and McNemar tests (Holm-corrected)

| ladder | from | to | MU_points_per_doubling | MU_ci | p_holm | significant |
|---|---|---|---|---|---|---|
| primary | qwen2.5-0.5b | qwen2.5-1.5b | 2.81 | [0.0, 5.6] | 0.0161 | True |
| primary | qwen2.5-1.5b | qwen2.5-3b | 31.83 | [26.4, 37.3] | 3.44e-21 | True |
| primary | qwen2.5-3b | qwen2.5-7b | 6.70 | [3.3, 10.2] | 0.000313 | True |
| primary | qwen2.5-7b | qwen2.5-14b | 20.08 | [15.3, 25.0] | 6.11e-06 | True |

## Curve fits: log-linear (H0) vs saturating power law (H1)

delta_aic = AIC(log-linear) − AIC(saturating); positive favours H1. frac_boot_favour_saturating = share of item-bootstrap replicates with delta_aic > 0.

| ladder | n_points | loglin_aic | sat_aic | delta_aic | delta_aicc | frac_boot_favour_saturating | sat_A_max | sat_alpha | N90_b | N90_lo | N90_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | 5 | -20.8 | -15.9 | -4.83 |  | 0 | 1 | 0.293 | 14.7 | 14.7 | 14.7 |

Per-domain and per-tier results are in the CSV files next to this summary (column `subset`). Figures: fig_scaling.png, fig_domains.png, fig_marginal_utility.png, fig_efficiency.png.
