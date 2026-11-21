# Evaluation summary

## Abstract placeholders (primary ladder, all items)

- [X] accuracy gain qwen2.5-0.5b → qwen2.5-3b: **36.6 points**
- [Y] accuracy gain qwen2.5-3b → qwen2.5-14b: **27.8 points**
- [Z] inference compute per response, qwen2.5-14b / qwen2.5-0.5b: **36.2×**

## Per-model metrics (all items, greedy run; variance from sampled runs)

| model_id | ladder | params_b | accuracy | ci | format_failure_rate | variance | mean_output_tokens | ter | U_points_per_tflop |
|---|---|---|---|---|---|---|---|---|---|
| gemma2-2b | comparator | 2.61 | 30.4 | [25.6, 35.4] | 20.3 | 0.0411 | 133 | 229 | 19.7 |
| phi3-mini | comparator | 3.82 | 44.0 | [38.9, 49.0] | 21.0 | 0.0906 | 209 | 211 | 15.1 |
| mistral-7b | comparator | 7.25 | 28.0 | [23.6, 32.5] | 15.3 | 0.0766 | 153 | 182 | 6.06 |
| gemma2-9b | comparator | 9.24 | 43.2 | [38.0, 48.3] | 18.0 | 0.0464 | 119 | 364 | 8.31 |
| qwen2.5-0.5b | primary | 0.49 | 9.1 | [6.2, 12.2] | 61.7 | 0.0506 | 160 | 56.9 | 28.7 |
| qwen2.5-1.5b | primary | 1.54 | 13.8 | [10.1, 17.7] | 70.0 | 0.065 | 157 | 87.7 | 14 |
| qwen2.5-3b | primary | 3.09 | 45.7 | [40.3, 51.1] | 21.7 | 0.0766 | 169 | 271 | 22.3 |
| qwen2.5-7b | primary | 7.61 | 54.4 | [49.1, 59.8] | 12.3 | 0.0597 | 162 | 336 | 11 |
| qwen2.5-14b | primary | 14.7 | 73.5 | [69.3, 77.6] | 1.3 | 0.0322 | 227 | 324 | 6.41 |
| llama3.1-8b | secondary | 8.03 | 49.2 | [44.0, 54.5] | 19.3 | 0.0599 | 185 | 266 | 8.78 |

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
