"""Command line: python -m llmeval <command> --config config.yaml

  validate      check the prompt matrix and the scorers against the ground truth (no models needed)
  run           inference + logging for the configured models (greedy and K sampled runs)
  score         extraction + scoring of every logged response; rubric items go to the Claude judge
  similarity    semantic divergence (Eq. 2) and BERTScore F1 (optional dependencies)
  analyze       metrics, curve fits, bootstrap CIs, McNemar/Holm, figures and summary.md
  human-export  blinded rubric grading sheets for two human graders
  human-import  read the filled sheets back and report inter-rater agreement
"""
import argparse
import os
import sys

import pandas as pd

from .config import load_config
from .items import load_items


def _models(cfg, names):
    ms = cfg.models
    if names:
        wanted = set(names.split(","))
        ms = [m for m in ms if m.id in wanted]
        missing = wanted - {m.id for m in ms}
        if missing:
            sys.exit(f"unknown model ids: {', '.join(sorted(missing))}")
    return ms


def _items(cfg, args):
    items = load_items(cfg.items_csv)
    if getattr(args, "items", None):
        prefixes = tuple(args.items.split(","))
        items = [it for it in items if it.item_id.startswith(prefixes)]
    if getattr(args, "limit", None):
        items = items[: args.limit]
    return items


def _scores(cfg):
    path = os.path.join(cfg.run_dir, "scores.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found; run the 'score' command first")
    return pd.read_csv(path)


def cmd_validate(cfg, args):
    from .validate import validate
    ok = validate(load_items(cfg.items_csv))
    sys.exit(0 if ok else 1)


def cmd_run(cfg, args):
    from .runner import run_model
    items = _items(cfg, args)
    for spec in _models(cfg, args.models):
        run_model(cfg, spec, items, which=args.runs)


def cmd_score(cfg, args):
    from .pipeline import score_all
    df = score_all(cfg, load_items(cfg.items_csv), _models(cfg, args.models), use_judge=not args.no_judge)
    if df.empty:
        sys.exit("no responses to score")
    g = df[df.run == "greedy"].groupby("model_id").agg(accuracy=("score", "mean"),
                                                         format_failures=("format_failure", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {os.path.join(cfg.run_dir, 'scores.csv')}")


def cmd_similarity(cfg, args):
    from .similarity import compute
    compute(cfg, _scores(cfg), load_items(cfg.items_csv), _models(cfg, args.models))


def cmd_analyze(cfg, args):
    from .analysis import analyze, headline
    from .report import write_report
    sim_path = os.path.join(cfg.run_dir, "similarity.csv")
    sim = pd.read_csv(sim_path) if os.path.exists(sim_path) else None
    a = cfg.analysis
    res = analyze(_scores(cfg), _models(cfg, args.models), sim, n_boot=a["bootstrap_resamples"],
                  alpha=a["alpha"], fit_boot=a.get("fit_bootstrap_resamples", 1000), seed=cfg.seed)
    out = os.path.join(cfg.run_dir, "analysis")
    write_report(res, headline(res, cfg.models, a.get("mid_model")), out)
    print(open(os.path.join(out, "summary.md")).read())


def cmd_human_export(cfg, args):
    from .human import export_sheets
    print("wrote", export_sheets(cfg, _scores(cfg), load_items(cfg.items_csv), n=args.n))


def cmd_human_import(cfg, args):
    from .human import import_sheets
    print(import_sheets(cfg, _scores(cfg)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="llmeval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--models", help="comma-separated model ids (default: all in config)")
    r.add_argument("--runs", choices=["all", "greedy", "samples"], default="all")
    r.add_argument("--items", help="comma-separated item-id prefixes, e.g. MATH-B,CODE")
    r.add_argument("--limit", type=int, help="only the first N items (smoke tests)")
    s = sub.add_parser("score")
    s.add_argument("--models")
    s.add_argument("--no-judge", action="store_true", help="skip the rubric judge (EXPL items stay unscored)")
    for name in ("similarity", "analyze"):
        sub.add_parser(name).add_argument("--models")
    he = sub.add_parser("human-export")
    he.add_argument("--n", type=int, default=100, help="number of responses to sample")
    sub.add_parser("human-import")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"validate": cmd_validate, "run": cmd_run, "score": cmd_score, "similarity": cmd_similarity,
     "analyze": cmd_analyze, "human-export": cmd_human_export, "human-import": cmd_human_import}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
