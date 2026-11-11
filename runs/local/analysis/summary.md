# Evaluation summary

## Per-model metrics (all items, greedy run; variance from sampled runs)

| model_id | ladder | params_b | accuracy | ci | format_failure_rate | variance | mean_output_tokens | ter | U_points_per_tflop |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | primary | 0.49 | 9.1 | [6.2, 12.2] | 61.7 | 0.0506 | 160 | 56.9 | 28.7 |
| llama3.1-8b | secondary | 8.03 | 49.2 | [44.0, 54.5] | 19.3 | 0.0602 | 185 | 266 | 8.78 |

## Marginal utility per doubling (Eq. 6) and McNemar tests (Holm-corrected)

_Not available: needs at least 2 scored models on the same ladder._

## Curve fits: log-linear (H0) vs saturating power law (H1)

delta_aic = AIC(log-linear) − AIC(saturating); positive favours H1. frac_boot_favour_saturating = share of item-bootstrap replicates with delta_aic > 0.

_Not available: needs at least 2 scored models on the same ladder._

Per-domain and per-tier results are in the CSV files next to this summary (column `subset`). Figures: fig_scaling.png, fig_domains.png, fig_marginal_utility.png, fig_efficiency.png.
