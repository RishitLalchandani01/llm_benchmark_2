"""Semantic divergence (Sec. III-E, metric 2, Eq. 2) and BERTScore F1 as its robustness check.

D = 1 - cos(E(response), E(reference)). Computed on the greedy run for items with a text reference:
the extracted answer vs the canonical answer for short-answer knowledge items (FACT), and the full
response vs the reference answer for explanation items (EXPL). Numeric and code items are scored
exactly and are left out. Needs the optional dependencies: pip install sentence-transformers bert-score
"""
import os

import numpy as np
import pandas as pd

from .extract import first_line
from .runner import load_responses, responses_path


def compute(cfg, scores: pd.DataFrame, items, models, progress=print) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    by_id = {it.item_id: it for it in items}
    rows = []
    for spec in models:
        resp = load_responses(responses_path(cfg.run_dir, spec.id))
        for (item_id, run), r in resp.items():
            it = by_id.get(item_id)
            if run != "greedy" or it is None or it.domain_code not in ("FACT", "EXPL"):
                continue
            if it.domain_code == "FACT":
                sub = scores[(scores.model_id == spec.id) & (scores.item_id == item_id) & (scores.run == run)]
                ext = sub.extracted.iloc[0] if len(sub) and isinstance(sub.extracted.iloc[0], str) else ""
                hyp, ref = first_line(ext), it.canonical_answer
            else:
                hyp, ref = r["text"], it.reference_answer
            rows.append({"model_id": spec.id, "item_id": item_id, "domain": it.domain_code,
                         "tier": it.tier, "hyp": hyp or "", "ref": ref})
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sc = cfg.similarity
    progress(f"[similarity] embedding {len(df)} pairs with {sc['embedding_model']}")
    enc = SentenceTransformer(sc["embedding_model"])
    h = enc.encode(df.hyp.tolist(), normalize_embeddings=True, batch_size=64)
    r = enc.encode(df.ref.tolist(), normalize_embeddings=True, batch_size=64)
    df["divergence"] = 1 - np.sum(h * r, axis=1)

    try:
        from bert_score import score as bert_score
        progress(f"[similarity] BERTScore with {sc['bertscore_model']}")
        _, _, f1 = bert_score(df.hyp.tolist(), df.ref.tolist(), model_type=sc["bertscore_model"], verbose=False)
        df["bertscore_f1"] = f1.numpy()
    except ImportError:
        progress("[similarity] bert-score not installed; skipping BERTScore")
    out = df.drop(columns=["hyp", "ref"])
    out.to_csv(os.path.join(cfg.run_dir, "similarity.csv"), index=False)
    return out
