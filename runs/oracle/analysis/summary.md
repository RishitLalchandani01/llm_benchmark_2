# Evaluation summary

## Per-model metrics (all items, greedy run; variance from sampled runs)

| model_id | ladder | params_b | accuracy | ci | format_failure_rate | variance | mean_output_tokens | ter | U_points_per_tflop |
|---|---|---|---|---|---|---|---|---|---|
| oracle-1b | primary | 1 | 83.3 | [79.0, 87.5] | 0.0 | 0 | 12 | 6.87e+03 | 756 |
| oracle-4b | primary | 4 | 83.3 | [79.0, 87.5] | 0.0 | 0 | 12 | 6.87e+03 | 189 |

## Marginal utility per doubling (Eq. 6) and McNemar tests (Holm-corrected)

| ladder | from | to | MU_points_per_doubling | MU_ci | p_holm | significant |
|---|---|---|---|---|---|---|
| primary | oracle-1b | oracle-4b | 0.00 | [0.0, 0.0] | 1 | False |

## Curve fits: log-linear (H0) vs saturating power law (H1)

delta_aic = AIC(log-linear) − AIC(saturating); positive favours H1. frac_boot_favour_saturating = share of item-bootstrap replicates with delta_aic > 0.

| ladder | n_points | loglin_aic | sat_aic | delta_aic | delta_aicc | sat_A_max | sat_alpha | N90_b | N90_lo | N90_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | 2 | -50.6 | -48.6 |  |  | 0.833 | 0.00333 | 1 | 1 | 1 |

Per-domain and per-tier results are in the CSV files next to this summary (column `subset`). Figures: fig_scaling.png, fig_domains.png, fig_marginal_utility.png, fig_efficiency.png.
