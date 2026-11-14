"""Stages 2-3: inference and logging (Sec. III-D).

Each model gets a primary greedy run (temperature 0, one response per item) and K sampled runs
(temperature 0.7, top-p 0.95, fixed seeds). Items are presented in a seeded random order, one
independent request each, so nothing carries over between items. Output: one JSONL file per model
under <run_dir>/responses/, appended as responses arrive, so an interrupted run resumes where it stopped.
"""
import json
import os
import platform
import random
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .backends import make_backend
from .prompting import TEMPLATE_VERSION, build_messages
from .log import log


def run_ids(k_samples: int, which: str) -> list:
    ids = []
    if which in ("all", "greedy"):
        ids.append("greedy")
    if which in ("all", "samples"):
        ids += [f"sample{k}" for k in range(1, k_samples + 1)]
    return ids


def responses_path(run_dir, model_id):
    return os.path.join(run_dir, "responses", f"{model_id}.jsonl")


def load_responses(path) -> dict:
    """{(item_id, run): record}, keeping the last successful record for each key."""
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("error") is None:
                    out[(r["item_id"], r["run"])] = r
    return out


def run_model(cfg, spec, items, which="all", progress=log):
    dec = cfg.decoding
    bspec = cfg.backends[spec.backend]
    backend = make_backend(bspec, items=items)
    path = responses_path(cfg.run_dir, spec.id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = load_responses(path)

    order = list(items)
    random.Random(cfg.seed + zlib.crc32(spec.id.encode())).shuffle(order)
    jobs = [(it, run) for run in run_ids(dec["k_samples"], which) for it in order
            if (it.item_id, run) not in done]
    progress(f"[{spec.id}] {len(jobs)} requests to send ({len(done)} already logged)")
    if not jobs:
        return

    hardware = spec.hardware or bspec.get("hardware") or f"{platform.machine()} {platform.processor()}".strip()
    lock = threading.Lock()

    def one(it, run):
        greedy = run == "greedy"
        k = 0 if greedy else int(run[len("sample"):])
        temperature = dec["greedy_temperature"] if greedy else dec["sample_temperature"]
        top_p = 1.0 if greedy else dec["sample_top_p"]
        seed = cfg.seed + k
        rec = {
            "model_id": spec.id, "item_id": it.item_id, "run": run, "seed": seed,
            "temperature": temperature, "top_p": top_p, "max_tokens": dec["max_tokens"],
            "backend": spec.backend, "model": spec.model, "model_version": spec.version,
            "precision": spec.precision, "hardware": hardware, "template": TEMPLATE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(), "error": None,
        }
        try:
            c = backend.generate(spec.model, build_messages(it, spec.system_in_user), temperature,
                                 top_p, dec["max_tokens"], seed, spec.extra_body)
            rec.update(text=c.text, input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                       latency_s=round(c.latency_s, 4), finish_reason=c.finish_reason,
                       served_model=c.served_model)
        except Exception as e:  # logged and retried on the next invocation
            rec["error"] = f"{type(e).__name__}: {e}"
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    workers = spec.concurrency or bspec.get("concurrency", 4)
    n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, it, run) for it, run in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result()["error"]:
                n_err += 1
                if n_err <= 3:
                    progress(f"[{spec.id}] error: {fut.result()['error']}")
            if i % 50 == 0 or i == len(jobs):
                progress(f"[{spec.id}] {i}/{len(jobs)} done, {n_err} errors")
