"""Human grading for the explanation rubric: blinded export for two independent graders, then import
with inter-rater agreement (Cohen's kappa) between the graders and between each grader and the judge.

Export writes, under <run_dir>/human/:
  grader_A.csv, grader_B.csv  one row per (response, key point); fill the 'present' column with 1 or 0
  key.csv                     maps anonymous response ids back to model/item/run (keep it from graders)
"""
import json
import os
import random

import pandas as pd


def export_sheets(cfg, scores: pd.DataFrame, items, n: int = 100, seed: int = None) -> str:
    by_id = {it.item_id: it for it in items}
    rub = scores[(scores.rule == "rubric") & (scores.run == "greedy")]
    rng = random.Random(cfg.seed if seed is None else seed)
    # Stratify by model, so every model contributes about equally to the sample.
    groups = [g.sample(frac=1, random_state=rng.randrange(2**31)) for _, g in rub.groupby("model_id")]
    picked, i = [], 0
    while len(picked) < min(n, len(rub)):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    out = os.path.join(cfg.run_dir, "human")
    os.makedirs(out, exist_ok=True)
    key, rows = [], []
    for r in picked:
        rid = f"R{rng.randrange(16**8):08x}"
        key.append({"response_id": rid, "model_id": r.model_id, "item_id": r.item_id, "run": r.run})
        it = by_id[r.item_id]
        for j, kp in enumerate(it.key_points, 1):
            rows.append({"response_id": rid, "item_id": r.item_id, "question": it.prompt,
                         "key_point_index": j, "key_point": kp, "response": r.judge_text, "present": ""})
    pd.DataFrame(key).to_csv(os.path.join(out, "key.csv"), index=False)
    sheet = pd.DataFrame(rows)
    for name in ("A", "B"):
        ids = sheet.response_id.unique().tolist()
        rng.shuffle(ids)
        order = {rid: k for k, rid in enumerate(ids)}
        (sheet.assign(_o=sheet.response_id.map(order)).sort_values(["_o", "key_point_index"])
         .drop(columns="_o").to_csv(os.path.join(out, f"grader_{name}.csv"), index=False))
    return out


def cohen_kappa(a, b) -> float:
    a, b = list(map(int, a)), list(map(int, b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def import_sheets(cfg, scores: pd.DataFrame) -> dict:
    d = os.path.join(cfg.run_dir, "human")
    key = pd.read_csv(os.path.join(d, "key.csv"))
    cols = ["response_id", "key_point_index", "present"]
    A = pd.read_csv(os.path.join(d, "grader_A.csv"))[cols].rename(columns={"present": "A"})
    B = pd.read_csv(os.path.join(d, "grader_B.csv"))[cols].rename(columns={"present": "B"})
    m = A.merge(B, on=["response_id", "key_point_index"]).merge(key, on="response_id")
    m = m.dropna(subset=["A", "B"])

    judge = scores[(scores.rule == "rubric")].set_index(["model_id", "item_id", "run"])["detail"]
    def judge_point(r):
        try:
            return int(json.loads(judge.loc[(r.model_id, r.item_id, r.run)])["key_points_present"][r.key_point_index - 1])
        except (KeyError, TypeError, IndexError):
            return None
    m["J"] = m.apply(judge_point, axis=1)
    mj = m.dropna(subset=["J"])

    report = {
        "n_responses": int(m.response_id.nunique()), "n_key_points": int(len(m)),
        "kappa_A_B": cohen_kappa(m.A, m.B),
        "kappa_judge_A": cohen_kappa(mj.J, mj.A) if len(mj) else None,
        "kappa_judge_B": cohen_kappa(mj.J, mj.B) if len(mj) else None,
        "raw_agreement_A_B": float((m.A == m.B).mean()),
    }
    per_resp = m.groupby(["model_id", "item_id", "run"]).agg(human_A=("A", "mean"), human_B=("B", "mean")).reset_index()
    per_resp["human_score"] = (per_resp.human_A + per_resp.human_B) / 2
    per_resp.to_csv(os.path.join(d, "human_scores.csv"), index=False)
    with open(os.path.join(d, "agreement.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
