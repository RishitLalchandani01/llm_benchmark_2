"""Write analysis tables, a markdown summary and figures to <run_dir>/analysis/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import loglinear, saturating  # noqa: E402

# Validated categorical slots 1-3 (fixed order) and recessive ink.
LADDER_COLOR = {"primary": "#2a78d6", "secondary": "#eb6834", "comparator": "#1baf7a"}
LADDER_MARKER = {"primary": "o", "secondary": "s", "comparator": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOMAIN_NAMES = {"MATH": "Mathematical reasoning", "CODE": "Code generation", "FACT": "Factual knowledge",
                "EXPL": "Scientific explanation", "INST": "Instruction following"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 5, "axes.axisbelow": True,
})


def _scaling_axis(ax, mm, fits, subset, label_comparators=True, legend=True):
    d = mm[mm.subset == subset]
    for lad in ("primary", "secondary", "comparator"):
        p = d[d.ladder == lad].sort_values("params_b")
        if p.empty:
            continue
        col = LADDER_COLOR[lad]
        y = 100 * p.accuracy
        err = [100 * (p.accuracy - p.accuracy_lo), 100 * (p.accuracy_hi - p.accuracy)]
        ax.errorbar(p.params_b, y, yerr=err, fmt=LADDER_MARKER[lad], color=col, ecolor=col, elinewidth=1,
                    capsize=0, markeredgecolor="white", markeredgewidth=0.8, label=f"{lad} ladder"
                    if lad != "comparator" else "compact comparators", zorder=3)
        f = fits[(fits.subset == subset) & (fits.ladder == lad)] if len(fits) else fits
        if lad != "comparator" and len(f):
            f = f.iloc[0]
            xs = np.geomspace(p.params_b.min(), p.params_b.max(), 200)
            if np.isfinite(f.get("sat_A_max", np.nan)):
                ax.plot(xs, 100 * saturating(xs, f.sat_A_max, f.sat_c, f.sat_alpha), color=col, lw=1.5, zorder=2)
            ax.plot(xs, 100 * loglinear(xs, f.loglin_a, f.loglin_b), color=col, lw=1, ls=(0, (3, 2)), zorder=2)
        if lad == "comparator" and label_comparators:
            for k, (_, r) in enumerate(p.iterrows()):  # alternate above/below to avoid collisions
                top = 100 * (r.accuracy_hi if k % 2 == 0 else r.accuracy_lo)
                ax.annotate(r.model_id, (r.params_b, top), xytext=(0, 3 if k % 2 == 0 else -3),
                            textcoords="offset points", fontsize=6, color=INK2, ha="center",
                            va="bottom" if k % 2 == 0 else "top")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("Accuracy (%)")
    if legend:
        ax.legend(loc="lower right", fontsize=7)


def fig_scaling(mm, fits, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    _scaling_axis(ax, mm, fits, "all")
    ax.set_title("Accuracy vs parameter count\nsolid: saturating fit · dashed: log-linear fit · bars: 95% CI",
                 fontsize=7, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_scaling.png"))
    plt.close(fig)


def fig_domains(mm, fits, out):
    subsets = ["all"] + [f"domain={d}" for d in DOMAIN_NAMES if f"domain={d}" in set(mm.subset)]
    cols = 3
    rows = int(np.ceil(len(subsets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.3 * rows), sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, subsets):
        _scaling_axis(ax, mm, fits, s, label_comparators=False, legend=(s == "all"))
        ax.set_title("All domains" if s == "all" else DOMAIN_NAMES[s.split("=")[1]], fontsize=8, loc="left")
    for k, ax in enumerate(axes.flat):
        if k >= len(subsets):
            ax.set_visible(False)
        if k % cols:
            ax.set_ylabel("")
        if k < len(subsets) - cols:
            ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_domains.png"))
    plt.close(fig)


def fig_mu(mu, out):
    if mu.empty:  # no ladder has >= 2 scored models yet
        return
    d = mu[mu.subset == "all"]
    lads = [l for l in ("primary", "secondary") if l in set(d.ladder)]
    if not lads:
        return
    fig, axes = plt.subplots(1, len(lads), figsize=(3.5 * len(lads), 2.4), squeeze=False)
    for ax, lad in zip(axes.flat, lads):
        p = d[d.ladder == lad].sort_values("N_from")
        x = np.arange(len(p))
        ax.bar(x, p.MU_points_per_doubling, width=0.6, color=LADDER_COLOR[lad], edgecolor="white", linewidth=1)
        ax.errorbar(x, p.MU_points_per_doubling, yerr=[p.MU_points_per_doubling - p.MU_lo,
                    p.MU_hi - p.MU_points_per_doubling], fmt="none", ecolor=INK2, elinewidth=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [f"{a:g}→{b:g}B" for a, b in zip(p.N_from, p.N_to)], fontsize=7)
        ax.set_ylabel("Accuracy gain per doubling (points)")
        ax.set_title(f"Marginal utility per doubling, {lad} ladder", fontsize=8, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_marginal_utility.png"))
    plt.close(fig)


def fig_efficiency(mm, out):
    d = mm[mm.subset == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, colname, ylabel, log in ((axes[0], "ter", "Accuracy points per 1k output tokens", False),
                                     (axes[1], "U_points_per_tflop", "Accuracy points per TFLOP", True)):
        for lad in ("primary", "secondary", "comparator"):
            p = d[d.ladder == lad].sort_values("params_b")
            if p.empty:
                continue
            ls = "-" if lad != "comparator" else "none"
            ax.plot(p.params_b, p[colname], marker=LADDER_MARKER[lad], ls=ls, color=LADDER_COLOR[lad],
                    markeredgecolor="white", markeredgewidth=0.8, label=lad)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Token efficiency ratio (Eq. 4)", fontsize=8, loc="left")
    axes[1].set_title("Cost-normalized utility (Eq. 5)", fontsize=8, loc="left")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_efficiency.png"))
    plt.close(fig)


def _md_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, str)(r[c]) if c in r and r[c] == r[c] else "" for c in cols) + " |")
    return "\n".join(lines)


def write_report(results: dict, head: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in results.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, "headline.json"), "w") as f:
        json.dump(head, f, indent=2, default=float)

    mm, mu, fits, mc = (results[k] for k in ("model_metrics", "marginal_utility", "curve_fits", "mcnemar"))
    fig_scaling(mm, fits, out_dir)
    fig_domains(mm, fits, out_dir)
    fig_mu(mu, out_dir)
    fig_efficiency(mm, out_dir)

    pct = lambda v: f"{100 * v:.1f}"
    num = lambda v: f"{v:.3g}" if isinstance(v, (int, float, np.number)) else str(v)
    a = mm[mm.subset == "all"].sort_values(["ladder", "params_b"]).copy()
    a["ci"] = a.apply(lambda r: f"[{100 * r.accuracy_lo:.1f}, {100 * r.accuracy_hi:.1f}]", axis=1)
    cols = ["model_id", "ladder", "params_b", "accuracy", "ci", "format_failure_rate", "variance",
            "mean_output_tokens", "ter", "U_points_per_tflop", "semantic_divergence", "bertscore_f1"]
    cols = [c for c in cols if c in a]
    md = ["# Evaluation summary", ""]
    if head:
        md += ["## Abstract placeholders (primary ladder, all items)", "",
               f"- [X] accuracy gain {head['smallest']} → {head['mid']}: **{head['X_points_small_to_mid']:.1f} points**",
               f"- [Y] accuracy gain {head['mid']} → {head['largest']}: **{head['Y_points_mid_to_large']:.1f} points**",
               f"- [Z] inference compute per response, {head['largest']} / {head['smallest']}: **{head['Z_cost_factor_small_to_large']:.1f}×**",
               ""]
    md += ["## Per-model metrics (all items, greedy run; variance from sampled runs)", "",
           _md_table(a, cols, {"accuracy": pct, "format_failure_rate": pct, "variance": num, "mean_output_tokens":
                               lambda v: f"{v:.0f}", "ter": num, "U_points_per_tflop": num,
                               "semantic_divergence": num, "bertscore_f1": num, "params_b": lambda v: f"{v:g}"}),
           "", "## Marginal utility per doubling (Eq. 6) and McNemar tests (Holm-corrected)", ""]
    if mu.empty:
        md += ["_Not available: needs at least 2 scored models on the same ladder._"]
    else:
        m = mu[mu.subset == "all"].merge(mc[mc.subset == "all"], left_on=["subset", "ladder", "from", "to"],
                                         right_on=["subset", "ladder", "smaller", "larger"], how="left")
        m["MU_ci"] = m.apply(lambda r: f"[{r.MU_lo:.1f}, {r.MU_hi:.1f}]", axis=1)
        md += [_md_table(m, ["ladder", "from", "to", "MU_points_per_doubling", "MU_ci", "p_holm", "significant"],
                         {"MU_points_per_doubling": lambda v: f"{v:.2f}", "p_holm": lambda v: f"{v:.3g}"})]
    md += ["", "## Curve fits: log-linear (H0) vs saturating power law (H1)", "",
           "delta_aic = AIC(log-linear) − AIC(saturating); positive favours H1. "
           "frac_boot_favour_saturating = share of item-bootstrap replicates with delta_aic > 0.", ""]
    f = fits[fits.subset == "all"] if len(fits) else fits
    if f.empty:
        md += ["_Not available: needs at least 2 scored models on the same ladder._"]
    else:
        md += [_md_table(f, [c for c in ["ladder", "n_points", "loglin_aic", "sat_aic", "delta_aic", "delta_aicc",
                                     "frac_boot_favour_saturating", "sat_A_max", "sat_alpha", "N90_b",
                                     "N90_lo", "N90_hi"] if c in f], {c: num for c in f.columns})]
    md += ["", "Per-domain and per-tier results are in the CSV files next to this summary "
           "(column `subset`). Figures: fig_scaling.png, fig_domains.png, fig_marginal_utility.png, fig_efficiency.png.", ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(md))
