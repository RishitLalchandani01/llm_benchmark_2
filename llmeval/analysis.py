"""Metrics (Sec. III-E) and statistical analysis (Sec. III-F).

Every analysis runs on the full matrix and again on each domain and each difficulty tier ("subsets").
Confidence intervals come from a nonparametric bootstrap over items (default 10,000 resamples); the
same item resamples are shared across models, so differences between models (marginal utility) get
paired intervals.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import binomtest


# ---------- curve fits (metric 5) ----------

def loglinear(N, a, b):
    return a + b * np.log(N)


def saturating(N, A_max, c, alpha):
    return A_max - c * np.power(N, -alpha)


def _aic(rss, n, k):
    """Gaussian-likelihood AIC and small-sample AICc; k counts the error variance as a parameter."""
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    return aic, aicc


def fit_curves(N, A) -> dict:
    """Fit A = a + b log N and A = A_max - c N^-alpha (N in billions) and compare them by AIC.
    delta_aic = AIC(log-linear) - AIC(saturating); positive favours the saturating curve (H1)."""
    N, A = np.asarray(N, float), np.asarray(A, float)
    n = len(N)
    out = {"n_points": n}
    b, a = np.polyfit(np.log(N), A, 1)
    rss_lin = float(np.sum((A - loglinear(N, a, b)) ** 2))
    out.update(loglin_a=a, loglin_b=b, loglin_rss=rss_lin)
    out["loglin_aic"], out["loglin_aicc"] = _aic(rss_lin, n, 3)

    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for alpha0 in (0.1, 0.3, 0.6, 1.0, 2.0):
            amax0 = min(1.0, A.max() + 0.02)
            c0 = max(amax0 - A[np.argmin(N)], 1e-3) * N.min() ** alpha0
            try:
                p, _ = curve_fit(saturating, N, A, p0=[amax0, c0, alpha0],
                                 bounds=([0, 0, 1e-4], [1.0, np.inf, 10.0]), maxfev=2000)
            except (RuntimeError, ValueError):
                continue
            rss = float(np.sum((A - saturating(N, *p)) ** 2))
            if best is None or rss < best[1]:
                best = (p, rss)
    if best is None:
        out.update(sat_A_max=np.nan, sat_c=np.nan, sat_alpha=np.nan, sat_rss=np.nan,
                   sat_aic=np.nan, sat_aicc=np.nan, delta_aic=np.nan, delta_aicc=np.nan)
        return out
    (amax, c, alpha), rss_sat = best
    out.update(sat_A_max=amax, sat_c=c, sat_alpha=alpha, sat_rss=rss_sat)
    out["sat_aic"], out["sat_aicc"] = _aic(rss_sat, n, 4)
    # With 3 points the 3-parameter saturating curve interpolates exactly; the comparison is uninformative.
    out["delta_aic"] = out["loglin_aic"] - out["sat_aic"] if n > 3 else np.nan
    out["delta_aicc"] = out["loglin_aicc"] - out["sat_aicc"]
    return out


def n90(N, A):
    """Smallest evaluated model reaching 90% of the accuracy of the largest model in its ladder."""
    N, A = np.asarray(N, float), np.asarray(A, float)
    target = 0.9 * A[np.argmax(N)]
    return float(N[A >= target].min())


# ---------- McNemar + Holm ----------

def mcnemar(x, y):
    """Exact McNemar test on paired binary outcomes x, y. Returns (b, c, p)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, float(p)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------- main analysis ----------

def _subsets(items_df):
    yield "all", items_df.index
    for d in sorted(items_df.domain.unique()):
        yield f"domain={d}", items_df.index[items_df.domain == d]
    for t in ("basic", "intermediate", "advanced"):
        if (items_df.tier == t).any():
            yield f"tier={t}", items_df.index[items_df.tier == t]


def _ci(boot, alpha):
    return np.nanpercentile(boot, 100 * alpha / 2, axis=0), np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)


def analyze(scores: pd.DataFrame, models, sim: pd.DataFrame = None, n_boot=10000, alpha=0.05,
            fit_boot=1000, seed=0, progress=print) -> dict:
    specs = {m.id: m for m in models if m.id in set(scores.model_id)}
    mids = list(specs)
    unscored = scores[scores.score.isna()]
    if len(unscored):
        progress(f"[analyze] warning: {len(unscored)} rows have no score (not judged, or judge errors) and are excluded")
    sc = scores.dropna(subset=["score"])

    greedy = sc[sc.run == "greedy"]
    S = greedy.pivot_table(index="item_id", columns="model_id", values="score")
    C = greedy.pivot_table(index="item_id", columns="model_id", values="correct")
    complete = S.dropna().index
    if len(complete) < len(S):
        progress(f"[analyze] warning: using the {len(complete)} items scored for every model (of {len(S)})")
    S, C = S.loc[complete, mids], C.loc[complete, mids]
    items_df = greedy.drop_duplicates("item_id").set_index("item_id").loc[complete, ["domain", "tier"]]

    g = greedy.set_index(["item_id", "model_id"])
    T_out = g.output_tokens.unstack().reindex(index=complete, columns=mids)
    T_in = g.input_tokens.unstack().reindex(index=complete, columns=mids)
    FF = g.format_failure.astype(float).unstack().reindex(index=complete, columns=mids)
    OVER = g.over_word_limit.astype(float).unstack().reindex(index=complete, columns=mids) \
        if "over_word_limit" in g else None

    samples = sc[sc.run.str.startswith("sample")]
    var_item = (samples.groupby(["item_id", "model_id"]).score.var(ddof=1).unstack()
                .reindex(index=complete, columns=mids)) if len(samples) else None

    Na = np.array([specs[m].n_active * 1e9 for m in mids])
    rng = np.random.default_rng(seed)
    ladders = {}
    for m in mids:
        if specs[m].ladder in ("primary", "secondary"):
            ladders.setdefault(specs[m].ladder, []).append(m)
    ladders = {k: sorted(v, key=lambda m: specs[m].params_b) for k, v in ladders.items()}

    res = {k: [] for k in ("model_metrics", "marginal_utility", "curve_fits", "mcnemar", "comparators")}
    for subset, idx in _subsets(items_df):
        s = S.loc[idx].to_numpy()
        Q = len(s)
        boot_idx = rng.integers(0, Q, size=(n_boot, Q))
        boot_A = s[boot_idx].mean(axis=1)                         # (n_boot, M)
        A = s.mean(axis=0)
        lo, hi = _ci(boot_A, alpha)
        t_out = T_out.loc[idx].to_numpy().mean(axis=0)
        t_in = T_in.loc[idx].to_numpy().mean(axis=0)
        ter = 100 * A / (t_out / 1000)                           # Eq. 4, accuracy points per 1k output tokens
        tflop = 2 * Na * (t_in + t_out) / 1e12                   # Eq. 5 denominator, TFLOPs per response
        U = 100 * A / tflop
        boot_tok = T_out.loc[idx].to_numpy()[boot_idx].mean(axis=1)
        boot_ter = 100 * boot_A / (boot_tok / 1000)
        ter_lo, ter_hi = _ci(boot_ter, alpha)
        if var_item is not None:
            v = var_item.loc[idx].to_numpy()
            var_m = np.nanmean(v, axis=0)                        # Eq. 3
            var_lo, var_hi = _ci(np.nanmean(v[boot_idx], axis=1), alpha)
        for j, m in enumerate(mids):
            row = {"subset": subset, "model_id": m, "ladder": specs[m].ladder, "family": specs[m].family,
                   "params_b": specs[m].params_b, "n_items": Q,
                   "accuracy": A[j], "accuracy_lo": lo[j], "accuracy_hi": hi[j],
                   "binary_accuracy": C.loc[idx, m].mean(),
                   "format_failure_rate": FF.loc[idx, m].mean(),
                   "mean_output_tokens": t_out[j], "mean_input_tokens": t_in[j],
                   "ter": ter[j], "ter_lo": ter_lo[j], "ter_hi": ter_hi[j],
                   "tflop_per_response": tflop[j], "U_points_per_tflop": U[j]}
            if var_item is not None:
                row.update(variance=var_m[j], variance_lo=var_lo[j], variance_hi=var_hi[j])
            if OVER is not None:
                ov = OVER.loc[idx, m].dropna()
                row["over_120_words_rate"] = ov.mean() if len(ov) else np.nan
            if sim is not None and len(sim):
                ss = sim[(sim.model_id == m) & sim.item_id.isin(idx)]
                if len(ss):
                    row["semantic_divergence"] = ss.divergence.mean()
                    if "bertscore_f1" in ss:
                        row["bertscore_f1"] = ss.bertscore_f1.mean()
            res["model_metrics"].append(row)

        col = {m: j for j, m in enumerate(mids)}
        for lname, lm in ladders.items():
            J = [col[m] for m in lm]
            N = np.array([specs[m].params_b for m in lm])
            # Marginal utility per doubling (Eq. 6), percentage points, with paired bootstrap CI
            pairs = []
            for a_, b_ in zip(range(len(lm) - 1), range(1, len(lm))):
                d = np.log2(N[b_] / N[a_])
                mu_boot = 100 * (boot_A[:, J[b_]] - boot_A[:, J[a_]]) / d
                mlo, mhi = _ci(mu_boot, alpha)
                xb, xc, p = mcnemar(C.loc[idx, lm[b_]].to_numpy(), C.loc[idx, lm[a_]].to_numpy())
                res["marginal_utility"].append({
                    "subset": subset, "ladder": lname, "from": lm[a_], "to": lm[b_],
                    "N_from": N[a_], "N_to": N[b_], "doublings": d,
                    "MU_points_per_doubling": 100 * (A[J[b_]] - A[J[a_]]) / d, "MU_lo": mlo, "MU_hi": mhi})
                pairs.append({"subset": subset, "ladder": lname, "smaller": lm[a_], "larger": lm[b_],
                              "larger_only_correct": xb, "smaller_only_correct": xc, "p": p})
            if pairs:
                for pr, pa in zip(pairs, holm([pr["p"] for pr in pairs])):
                    pr["p_holm"] = pa
                    pr["significant"] = pa < alpha
                res["mcnemar"] += pairs

            if len(lm) >= 2:
                fit = fit_curves(N, A[J])
                fit.update(subset=subset, ladder=lname, N90_b=n90(N, A[J]))
                # Bootstrap the model comparison and N90 over item resamples
                nb = min(fit_boot, n_boot)
                deltas, n90s = [], []
                for r in range(nb):
                    Ab = boot_A[r, J]
                    n90s.append(n90(N, Ab))
                    if len(lm) >= 4:  # with 3 points the saturating fit has no residual df
                        deltas.append(fit_curves(N, Ab)["delta_aic"])
                q = np.nanpercentile(n90s, [100 * alpha / 2, 50, 100 * (1 - alpha / 2)])
                fit.update(N90_lo=q[0], N90_median=q[1], N90_hi=q[2])
                if deltas:
                    dl = np.array(deltas, float)
                    fit["frac_boot_favour_saturating"] = float(np.nanmean(dl > 0))
                res["curve_fits"].append(fit)

        # Compact comparators vs the primary ladder's fitted curves
        prim = next((f for f in res["curve_fits"] if f["subset"] == subset and f["ladder"] == "primary"), None)
        for m in mids:
            if specs[m].ladder != "comparator" or prim is None:
                continue
            N = specs[m].params_b
            pred_sat = saturating(N, prim["sat_A_max"], prim["sat_c"], prim["sat_alpha"])
            pred_lin = loglinear(N, prim["loglin_a"], prim["loglin_b"])
            res["comparators"].append({
                "subset": subset, "model_id": m, "params_b": N, "accuracy": A[col[m]],
                "primary_curve_pred_saturating": pred_sat, "primary_curve_pred_loglinear": pred_lin,
                "residual_vs_saturating_points": 100 * (A[col[m]] - pred_sat),
                "above_ci_of_prediction": bool(lo[col[m]] > pred_sat)})

    return {k: pd.DataFrame(v) for k, v in res.items()}


def headline(results: dict, models, mid_model: str = None) -> dict:
    """Numbers for the abstract placeholders [X], [Y], [Z] on the primary ladder."""
    mm = results["model_metrics"]
    p = mm[(mm.subset == "all") & (mm.ladder == "primary")].sort_values("params_b")
    if len(p) < 3:
        return {}
    small, large = p.iloc[0], p.iloc[-1]
    if mid_model and mid_model in set(p.model_id):
        mid = p[p.model_id == mid_model].iloc[0]
    else:  # the model closest to the geometric midpoint of the ladder
        gm = np.sqrt(small.params_b * large.params_b)
        mid = p.iloc[int(np.argmin(np.abs(np.log(p.params_b / gm))))]
    return {
        "smallest": small.model_id, "mid": mid.model_id, "largest": large.model_id,
        "X_points_small_to_mid": 100 * (mid.accuracy - small.accuracy),
        "Y_points_mid_to_large": 100 * (large.accuracy - mid.accuracy),
        "Z_cost_factor_small_to_large": large.tflop_per_response / small.tflop_per_response,
        "cost_factor_mid_to_large": large.tflop_per_response / mid.tflop_per_response,
    }
