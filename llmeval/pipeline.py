"""Score every logged response and write <run_dir>/scores.csv (one row per model x item x run)."""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .judge import Judge
from .runner import load_responses, responses_path
from .scoring import score_response


class _ScoreCache:
    """Persisted cache of deterministic scores, so re-scoring does not re-execute model code."""

    def __init__(self, path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.data[r["key"]] = r["row"]

    @staticmethod
    def key(item, text):
        return hashlib.sha256(f"{item.item_id}\x00{item.scoring_rule}\x00{text}".encode()).hexdigest()

    def get_or_score(self, item, text):
        k = self.key(item, text)
        if k not in self.data:
            row = score_response(item, text)
            with self.lock:
                self.data[k] = row
                with open(self.path, "a") as f:
                    f.write(json.dumps({"key": k, "row": row}, ensure_ascii=False) + "\n")
        return dict(self.data[k])


def score_all(cfg, items, models, use_judge=True, progress=print) -> pd.DataFrame:
    by_id = {it.item_id: it for it in items}
    cache = _ScoreCache(os.path.join(cfg.run_dir, "score_cache.jsonl"))
    records = []
    for spec in models:
        resp = load_responses(responses_path(cfg.run_dir, spec.id))
        if not resp:
            progress(f"[{spec.id}] no responses logged, skipped")
            continue
        progress(f"[{spec.id}] scoring {len(resp)} responses")
        keys = [k for k in resp if k[0] in by_id]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            rows = list(pool.map(lambda k: cache.get_or_score(by_id[k[0]], resp[k]["text"]), keys))
        for (item_id, run), row in zip(keys, rows):
            r = resp[(item_id, run)]
            it = by_id[item_id]
            row.update(model_id=spec.id, item_id=item_id, run=run, domain=it.domain_code, tier=it.tier,
                       input_tokens=r.get("input_tokens"), output_tokens=r.get("output_tokens"),
                       latency_s=r.get("latency_s"), finish_reason=r.get("finish_reason"))
            records.append(row)

    rubric = [r for r in records if r["rule"] == "rubric"]
    if rubric and use_judge:
        judge = Judge(cfg.judge, os.path.join(cfg.run_dir, "judge_cache.jsonl"))
        verdicts = judge.grade_many([(by_id[r["item_id"]], r["judge_text"]) for r in rubric], progress)
        n_err = 0
        for r, v in zip(rubric, verdicts):
            if v["error"]:
                n_err += 1
                r["judge_error"] = v["error"]
                continue
            frac = sum(v["present"]) / len(v["present"])
            r["score_lenient"] = frac
            r["score"] = 0.0 if r["format_failure"] else frac
            r["detail"] = {"key_points_present": v["present"]}
        if n_err:
            progress(f"[judge] {n_err} responses unscored (see judge_error); grade them by hand or re-run score")

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["correct"] = df["score"].map(lambda s: None if pd.isna(s) else float(s >= 1 - 1e-9))
    df["detail"] = df["detail"].map(lambda d: json.dumps(d, default=str))
    df = df.sort_values(["model_id", "item_id", "run"]).reset_index(drop=True)
    df.to_csv(os.path.join(cfg.run_dir, "scores.csv"), index=False)
    return df
