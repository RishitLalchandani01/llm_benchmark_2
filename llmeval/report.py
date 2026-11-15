"""Write analysis tables, a markdown summary and figures to <run_dir>/analysis/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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


# ---------- shared figure helpers ----------

LADDER_LABEL = {"primary": "primary ladder", "secondary": "secondary ladder", "comparator": "compact comparators"}
SIZE_AXIS_LABEL = "Model size (billions of parameters, log scale)"
NICE_TICKS = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000,
              1500, 2000, 3000, 5000, 7000, 10000]
LABEL_STROKE = [patheffects.withStroke(linewidth=2.5, foreground="white")]


def _plain_log_ticks(axis, lo, hi, suffix=""):
    """Log-scale ticks at round values (0.5, 1, 2, 3, 5, 7, 10, ...) printed as plain numbers, no 10^n."""
    ticks = [t for t in NICE_TICKS if lo <= t <= hi]
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_minor_locator(mticker.NullLocator())
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}{suffix}"))


def _size_axis(ax, params_b):
    """Shared model-size x axis: log scale, padded limits, ticks like 0.5B, 1B, 2B ... 15B."""
    lo, hi = min(params_b) / 1.3, max(params_b) * 1.3
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    _plain_log_ticks(ax.xaxis, lo, hi, suffix="B")
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(SIZE_AXIS_LABEL)


def _point_label(r, value_text):
    return f"{r.model_id}\n{r.params_b:g}B · {value_text}"


def _legend_handles(ladders, connected=False, fit_styles=False):
    """Line2D proxies for a single figure-level legend."""
    h = [Line2D([], [], color=LADDER_COLOR[l], marker=LADDER_MARKER[l], markeredgecolor="white",
                markeredgewidth=0.8, markersize=7, ls="-" if connected and l != "comparator" else "none",
                label=LADDER_LABEL[l]) for l in ("primary", "secondary", "comparator") if l in ladders]
    if fit_styles:
        h += [Line2D([], [], color=INK2, lw=1.5, label="saturating power-law fit"),
              Line2D([], [], color=INK2, lw=1, ls=(0, (3, 2)), label="log-linear fit"),
              Line2D([], [], color=INK2, marker="|", markersize=9, ls="none", label="95% CI")]
    return h


def _segments(ax):
    """Every drawn line on the axes as display-space segments (x0, y0, x1, y1): fitted curves,
    connecting lines and error bars. Marker-only lines are skipped (markers are handled separately)."""
    segs, ids = [], []

    def add(xy):
        segs.append(np.hstack([xy[:-1], xy[1:]]))
        ids.append(np.full(len(xy) - 1, len(ids)))

    for ln in ax.get_lines():
        if ln.get_linestyle() in ("None", "none", "", " "):
            continue
        add(ax.transData.transform(np.column_stack([np.asarray(a, float) for a in ln.get_data()])))
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            for sg in coll.get_segments():           # each error bar is its own obstacle
                if len(sg) >= 2:
                    add(coll.get_transform().transform(sg))
    if not segs:
        return np.empty((0, 4)), np.empty(0, int)
    segs, ids = np.vstack(segs), np.concatenate(ids)
    ok = np.isfinite(segs).all(axis=1)
    return segs[ok], ids[ok]


def _segments_hitting(S, r):
    """Boolean mask: which segments in S (N x 4) cross rectangle r = (x0, y0, x1, y1). Liang-Barsky clipping."""
    hit = np.zeros(len(S), bool)
    if not len(S):
        return hit
    x0, y0, x1, y1 = S.T
    m = ((np.maximum(x0, x1) >= r[0]) & (np.minimum(x0, x1) <= r[2])
         & (np.maximum(y0, y1) >= r[1]) & (np.minimum(y0, y1) <= r[3]))
    if not m.any():
        return hit
    x0, y0, x1, y1 = x0[m], y0[m], x1[m], y1[m]
    dx, dy = x1 - x0, y1 - y0
    t0, t1, ok = np.zeros_like(dx), np.ones_like(dx), np.ones_like(dx, bool)
    for pk, qk in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        zero = pk == 0
        ok &= ~(zero & (qk < 0))
        t = qk / np.where(zero, 1, pk)
        t0 = np.where(~zero & (pk < 0), np.maximum(t0, t), t0)
        t1 = np.where(~zero & (pk > 0), np.minimum(t1, t), t1)
    hit[m] = ok & (t0 <= t1)
    return hit


def _segments_crossing(S, seg):
    """Boolean mask: which segments in S (N x 4) properly intersect the single segment seg (x0, y0, x1, y1)."""
    if not len(S):
        return np.zeros(0, bool)
    ax_, ay, bx, by = seg
    cx, cy, dx, dy = S.T

    def orient(px_, py, qx, qy, rx, ry):
        return np.sign((qx - px_) * (ry - py) - (qy - py) * (rx - px_))
    return ((orient(ax_, ay, bx, by, cx, cy) * orient(ax_, ay, bx, by, dx, dy) < 0)
            & (orient(cx, cy, dx, dy, ax_, ay) * orient(cx, cy, dx, dy, bx, by) < 0))


def _rects_hitting(R, r):
    if not len(R):
        return 0
    R = np.asarray(R)
    return int(((R[:, 0] < r[2]) & (R[:, 2] > r[0]) & (R[:, 1] < r[3]) & (R[:, 3] > r[1])).sum())


# Placement costs. Overlapping another label or crossing another leader is worst; then covering a dot or the
# point's own error bar; then the label crossing a line; a thin leader crossing a line costs little. A small
# cost per step outwards and for needing a leader keeps labels close to their dots when space allows.
HIT_LABEL, HIT_MARKER, HIT_LINE, HIT_LEADER_LINE, HIT_OUTSIDE = 20, 10, 3, 1, 1000
COST_PER_RING, COST_LEADER = 0.5, 1


def _place_labels(ax, points, fontsize=7, rings=(7, 13, 20, 30, 42, 56, 72, 90), leader_after=12, pad=2):
    """Label every point without overlaps. Call after the axis limits and the figure layout are final.

    points: list of (x, y, text, color) in data coordinates. For each point, candidate positions are tried
    in order: right, left, above, below, the four diagonals, then in-between angles, at increasing distances
    (`rings`, in points). Each candidate is scored by what it collides with (markers, the point's own error bar, fitted
    and connecting lines, other error bars, labels and leader lines already placed, the axes edge) plus a
    small cost for distance; the lowest score wins, earlier candidates winning ties. Labels whose nearest
    edge ends up more than `leader_after` points from the dot get a thin leader line in the dot's color.
    """
    if not points:
        return
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0                                     # display pixels per point
    segs, seg_ids = _segments(ax)
    box = ax.get_window_extent(renderer)
    inner = (box.x0 + pad, box.y0 + pad, box.x1 - pad, box.y1 - pad)
    disp = ax.transData.transform([(x, y) for x, y, _, _ in points])
    mr = (plt.rcParams["lines.markersize"] / 2 + 1.5) * px  # marker half-size incl. white edge
    markers = [(dx - mr, dy - mr, dx + mr, dy + mr) for dx, dy in disp]
    labels, leaders = [], np.empty((0, 4))                  # placed label boxes and leader segments

    # right, left, above, below, diagonals, then in-between angles (give slanted leaders in crowded spots)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, .5), (-1, .5), (1, -.5), (-1, -.5), (.5, 1), (-.5, 1), (.5, -1), (-.5, -1)]
    order = sorted(range(len(points)), key=lambda i: (disp[i][0], disp[i][1]))
    for i in order:
        x, y, text, color = points[i]
        cx, cy = disp[i]
        others = [m for k, m in enumerate(markers) if k != i]
        # the point's own error bar: a vertical segment group through the dot
        own_bar = set(seg_ids[(np.abs(segs[:, 0] - cx) < 0.5) & (np.abs(segs[:, 2] - cx) < 0.5)
                              & (np.minimum(segs[:, 1], segs[:, 3]) <= cy)
                              & (np.maximum(segs[:, 1], segs[:, 3]) >= cy)].tolist()) if len(segs) else set()
        t = ax.text(0, 0, text, fontsize=fontsize, fontweight="bold")
        tb = t.get_window_extent(renderer)
        t.remove()
        w, h = tb.width + 2 * pad, tb.height + 2 * pad
        best = None
        for ri, ring in enumerate(rings):
            for sx, sy in dirs:
                ox = sx * (ring * px + w / 2)                # box centre offset from the dot, pixels
                oy = sy * (ring * px + h / 2)
                r = (cx + ox - w / 2, cy + oy - h / 2, cx + ox + w / 2, cy + oy + h / 2)
                gap = np.hypot(max(r[0] - cx, cx - r[2], 0), max(r[1] - cy, cy - r[3], 0))
                leader = gap / px > leader_after
                inside = r[0] >= inner[0] and r[1] >= inner[1] and r[2] <= inner[2] and r[3] <= inner[3]
                crossed = set(np.unique(seg_ids[_segments_hitting(segs, r)]).tolist())
                score = (COST_PER_RING * ri
                         + HIT_LINE * len(crossed - own_bar) + HIT_MARKER * len(crossed & own_bar)
                         + HIT_LABEL * int(_segments_hitting(leaders, r).sum())
                         + HIT_MARKER * _rects_hitting(others, r)
                         + HIT_LABEL * _rects_hitting(labels, r)
                         + (0 if inside else HIT_OUTSIDE))
                if leader:
                    score += COST_LEADER
                    # A vertical leader would run along the point's own error bar and read as part of the CI.
                    if sx == 0:
                        score += HIT_MARKER
                    # The leader must not cut through labels, dots, lines or error bars. It starts at the
                    # marker's edge so the point's own error bar (which passes through the dot) is not counted.
                    L = np.hypot(ox, oy)
                    seg = np.array([[cx + ox / L * mr, cy + oy / L * mr, cx + ox, cy + oy]])
                    score += (HIT_LABEL * sum(bool(_segments_hitting(seg, b)[0]) for b in labels)
                              + HIT_MARKER * sum(bool(_segments_hitting(seg, b)[0]) for b in others)
                              + HIT_LEADER_LINE * len(np.unique(seg_ids[_segments_crossing(segs, seg[0])]))
                              + HIT_LABEL * int(_segments_crossing(leaders, seg[0]).sum()))
                if best is None or score < best[0]:
                    best = (score, sx, ox, oy, r, leader)
        _, sx, ox, oy, r, leader = best
        ax.annotate(text, (x, y), xytext=(ox / px, oy / px), textcoords="offset points",
                    ha="center", va="center", multialignment="left" if sx > 0 else "right" if sx < 0 else "center",
                    fontsize=fontsize, fontweight="bold", color=color, path_effects=LABEL_STROKE, zorder=6,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=1, shrinkB=mr / px)
                    if leader else None)
        labels.append(r)
        if leader:
            leaders = np.vstack([leaders, [cx, cy, cx + ox, cy + oy]])


def _reserve_top(fig, inches):
    """tight_layout leaving `inches` free at the top for the super-title and the shared legend."""
    fig.tight_layout(rect=(0, 0, 1, 1 - inches / fig.get_figheight()))


# ---------- accuracy vs size ----------

def _scaling_axis(ax, mm, fits, subset):
    """Draw one accuracy-vs-size panel. Returns the label specs for _place_labels."""
    d = mm[mm.subset == subset]
    labels = []
    for lad in ("primary", "secondary", "comparator"):
        p = d[d.ladder == lad].sort_values("params_b")
        if p.empty:
            continue
        col = LADDER_COLOR[lad]
        y = 100 * p.accuracy
        err = [100 * (p.accuracy - p.accuracy_lo), 100 * (p.accuracy_hi - p.accuracy)]
        ax.errorbar(p.params_b, y, yerr=err, fmt=LADDER_MARKER[lad], color=col, ecolor=col, elinewidth=1,
                    capsize=0, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        f = fits[(fits.subset == subset) & (fits.ladder == lad)] if len(fits) else fits
        if lad != "comparator" and len(f):
            f = f.iloc[0]
            xs = np.geomspace(p.params_b.min(), p.params_b.max(), 200)
            if np.isfinite(f.get("sat_A_max", np.nan)):
                ax.plot(xs, 100 * saturating(xs, f.sat_A_max, f.sat_c, f.sat_alpha), color=col, lw=1.5, zorder=2)
            ax.plot(xs, 100 * loglinear(xs, f.loglin_a, f.loglin_b), color=col, lw=1, ls=(0, (3, 2)), zorder=2)
        labels += [(r.params_b, 100 * r.accuracy, _point_label(r, f"{100 * r.accuracy:.1f}%"), col)
                   for _, r in p.iterrows()]
    _size_axis(ax, d.params_b)
    ax.set_ylabel("Accuracy (%)")
    return labels


def fig_scaling(mm, fits, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = _scaling_axis(ax, mm, fits, "all")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.06 * (y1 - y0), y1 + 0.06 * (y1 - y0))
    ax.set_title("Accuracy vs model size\nsolid: saturating fit · dashed: log-linear fit · bars: 95% CI",
                 fontsize=9, color=INK2, loc="left")
    ladders = set(mm[mm.subset == "all"].ladder)
    fig.legend(handles=_legend_handles(ladders), loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.995))
    _reserve_top(fig, 0.35)
    _place_labels(ax, labels, fontsize=7)
    fig.savefig(os.path.join(out, "fig_scaling.png"))
    plt.close(fig)


def fig_domains(mm, fits, out):
    subsets = ["all"] + [f"domain={d}" for d in DOMAIN_NAMES if f"domain={d}" in set(mm.subset)]
    cols = 3
    rows = int(np.ceil(len(subsets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 6 * rows), sharey=True, squeeze=False)
    panels = []
    for ax, s in zip(axes.flat, subsets):
        labels = _scaling_axis(ax, mm, fits, s)
        ax.set_ylim(-12, 112)
        ax.set_title("All domains" if s == "all" else DOMAIN_NAMES[s.split("=")[1]], fontsize=10, loc="left",
                     fontweight="bold")
        panels.append((ax, labels))
    for k, ax in enumerate(axes.flat):
        if k >= len(subsets):
            ax.set_visible(False)
        if k % cols:
            ax.set_ylabel("")
    fig.suptitle("Accuracy vs model size by domain (greedy run; bars: 95% CI)", fontsize=12, color=INK2,
                 x=0.01, ha="left", y=0.995)
    ladders = set(mm[mm.subset.isin(subsets)].ladder)
    fig.legend(handles=_legend_handles(ladders, fit_styles=True), loc="upper center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 1 - 0.3 / fig.get_figheight()))
    _reserve_top(fig, 0.7)
    for ax, labels in panels:
        _place_labels(ax, labels, fontsize=7)
    fig.savefig(os.path.join(out, "fig_domains.png"), dpi=150)
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = (
        (axes[0], "ter", "Accuracy points per 1,000 output tokens (higher = more efficient)",
         "Token efficiency ratio (Eq. 4)", False),
        (axes[1], "U_points_per_tflop", "Accuracy points per TFLOP of compute (higher = more efficient)",
         "Cost-normalized utility (Eq. 5), log y axis", True),
    )
    placed = []
    for ax, colname, ylabel, title, log in panels:
        labels = []
        for lad in ("primary", "secondary", "comparator"):
            p = d[d.ladder == lad].sort_values("params_b")
            if p.empty:
                continue
            ls = "-" if lad != "comparator" else "none"
            ax.plot(p.params_b, p[colname], marker=LADDER_MARKER[lad], ls=ls, color=LADDER_COLOR[lad],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            labels += [(r.params_b, r[colname], _point_label(r, f"{r[colname]:.1f}" if r[colname] >= 1
                                                                  else f"{r[colname]:.3g}"), LADDER_COLOR[lad])
                       for _, r in p.iterrows()]
        _size_axis(ax, d.params_b)
        v = d[colname].dropna()
        if log:
            ax.set_yscale("log")
            lo, hi = v.min() / 1.3, v.max() * 1.3
            ax.set_ylim(lo, hi)
            _plain_log_ticks(ax.yaxis, lo, hi)
        else:
            span = (v.max() - v.min()) or 1
            ax.set_ylim(v.min() - 0.12 * span, v.max() + 0.12 * span)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        placed.append((ax, labels))
    fig.suptitle("Efficiency vs model size (greedy run, all domains)", fontsize=12, color=INK2,
                 x=0.01, ha="left", y=0.995)
    fig.legend(handles=_legend_handles(set(d.ladder), connected=True), loc="upper center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1 - 0.3 / fig.get_figheight()))
    _reserve_top(fig, 0.7)
    for ax, labels in placed:
        _place_labels(ax, labels, fontsize=7.5)
    fig.savefig(os.path.join(out, "fig_efficiency.png"), dpi=150)
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
