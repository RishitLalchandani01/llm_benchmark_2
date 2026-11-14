"""Fixed-prompt LLM judge for the scientific-explanation rubric (Sec. III-E, metric 1).

The judge is a Claude model, outside every evaluated family, which avoids self-preference bias.
It sees the question, the key points and the response, never the model's identity, and marks each key
point present or absent. Item score = fraction of key points present. Verdicts are cached in
<run_dir>/judge_cache.jsonl keyed by (judge model, prompt version, item, response text), so re-scoring
costs nothing. With batch=True the uncached requests go through the Message Batches API (50% price,
results usually within an hour, at most 24 h); the batch id is saved so an interrupted run resumes it.
The judge model is held fixed: no automatic fallback to another model on a refusal; a
refused or failed item is left unscored and flagged for human grading instead.
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel
from .log import log

JUDGE_PROMPT_VERSION = "rubric-v1"

JUDGE_SYSTEM = (
    "You grade answers to science explanation questions against a fixed rubric of key points. "
    "For each key point, decide whether the response states it, in any wording, correctly and "
    "unambiguously. A key point that is only hinted at, stated incorrectly, or contradicted elsewhere "
    "in the response is absent. Judge content only: ignore style, length, fluency and confidence, and "
    "do not give credit for correct statements that are not on the rubric."
)

JUDGE_TEMPLATE = """<question>
{question}
</question>

<key_points>
{points}
</key_points>

<response>
{response}
</response>

Return one judgement per key point, in order, with its 1-based index, a one-sentence rationale that quotes or paraphrases the relevant part of the response, and whether the point is present."""


class PointJudgement(BaseModel):
    index: int
    rationale: str
    present: bool


class RubricVerdict(BaseModel):
    points: list[PointJudgement]


def cache_key(model, item_id, text):
    return hashlib.sha256(f"{model}\x00{JUDGE_PROMPT_VERSION}\x00{item_id}\x00{text}".encode()).hexdigest()


class Judge:
    def __init__(self, cfg: dict, cache_path: str):
        self.model = cfg["model"]
        self.effort = cfg["effort"]
        self.max_tokens = cfg["max_tokens"]
        self.concurrency = cfg["concurrency"]
        self.client = anthropic.Anthropic()
        self.cache_path = cache_path
        self.cache = {}
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("error") is None:
                        self.cache[r["key"]] = r

    def _call(self, item, text):
        points = "\n".join(f"{i}. {p}" for i, p in enumerate(item.key_points, 1))
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
                question=item.prompt, points=points, response=text)}],
            output_format=RubricVerdict,
        )
        _check_refusal(resp)
        return {**_interpret(item, resp.parsed_output), "request_id": resp._request_id}

    def _new_record(self, item, text):
        rec = {"key": cache_key(self.model, item.item_id, text), "item_id": item.item_id,
               "judge_model": self.model, "prompt_version": JUDGE_PROMPT_VERSION, "error": None}
        if not text.strip():
            rec.update(present=[False] * len(item.key_points), rationales=["empty response"] * len(item.key_points))
        return rec

    def _store(self, rec):
        with self.lock:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["error"] is None:
                self.cache[rec["key"]] = rec
        return rec

    def grade(self, item, text) -> dict:
        key = cache_key(self.model, item.item_id, text)
        if key in self.cache:
            return self.cache[key]
        rec = self._new_record(item, text)
        if "present" not in rec:
            try:
                rec.update(self._call(item, text))
            except (anthropic.APIError, RuntimeError, ValueError) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
        return self._store(rec)

    def grade_many(self, pairs, progress=log, batch=False):
        """pairs: list of (item, text). Returns verdicts in the same order."""
        todo = sum(cache_key(self.model, it.item_id, t) not in self.cache for it, t in pairs)
        progress(f"[judge] {len(pairs)} rubric responses, {todo} not yet cached ({self.model})")
        if not batch:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                return list(pool.map(lambda p: self.grade(*p), pairs))
        failed = self._grade_batch(pairs, progress)
        return [self.cache.get(cache_key(self.model, it.item_id, t))
                or failed.get(cache_key(self.model, it.item_id, t))
                or {"error": "not judged (missing from batch results)"} for it, t in pairs]

    # ---------- Message Batches API ----------

    def _batch_params(self, item, text):
        points = "\n".join(f"{i}. {p}" for i, p in enumerate(item.key_points, 1))
        return {"model": self.model, "max_tokens": self.max_tokens, "system": JUDGE_SYSTEM,
                "output_config": {"effort": self.effort,
                                  "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                "messages": [{"role": "user", "content": JUDGE_TEMPLATE.format(
                    question=item.prompt, points=points, response=text)}]}

    def _grade_batch(self, pairs, progress):
        """Judge every uncached pair through one or more batches. Returns {key: error record} for failures."""
        state_path = os.path.join(os.path.dirname(self.cache_path), "judge_batch.json")
        by_key = {cache_key(self.model, it.item_id, t): (it, t) for it, t in pairs}
        failed = {}
        if os.path.exists(state_path):  # an earlier run submitted a batch: collect it before submitting more
            with open(state_path) as f:
                state = json.load(f)
            progress(f"[judge] resuming batch {state['batch_id']} submitted at {state['submitted_at']}")
            failed.update(self._collect(state, by_key, progress))
            os.remove(state_path)

        todo = {}
        for key, (it, t) in by_key.items():
            if key in self.cache or key in failed:
                continue
            rec = self._new_record(it, t)
            if "present" in rec:  # empty response: no API call needed
                self._store(rec)
            else:
                todo[key] = (it, t)
        if not todo:
            return failed

        try:
            b = self.client.messages.batches.create(requests=[
                {"custom_id": key, "params": self._batch_params(it, t)} for key, (it, t) in todo.items()])
        except anthropic.APIError as e:
            progress(f"[judge] batch submission failed: {type(e).__name__}: {e}")
            for key, (it, t) in todo.items():
                rec = self._new_record(it, t)
                rec["error"] = f"{type(e).__name__}: {e}"
                failed[key] = self._store(rec)
            return failed
        state = {"batch_id": b.id, "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "model": self.model, "prompt_version": JUDGE_PROMPT_VERSION, "keys": list(todo)}
        with open(state_path, "w") as f:
            json.dump(state, f)
        progress(f"[judge] submitted batch {b.id} with {len(todo)} requests; id saved to {state_path}")
        failed.update(self._collect(state, todo, progress))
        os.remove(state_path)
        return failed

    def _collect(self, state, todo, progress):
        """Wait for a batch to end, then store its verdicts. todo maps custom_id (cache key) -> (item, text)."""
        bid = state["batch_id"]
        try:
            while True:
                b = self.client.messages.batches.retrieve(bid)
                c = b.request_counts
                if b.processing_status == "ended":
                    break
                progress(f"[judge] batch {bid}: {b.processing_status}, {c.processing} processing, "
                         f"{c.succeeded} succeeded, {c.errored} errored; checking again in "
                         f"{BATCH_POLL_SECONDS} s")
                time.sleep(BATCH_POLL_SECONDS)
        except KeyboardInterrupt:
            progress(f"[judge] stopped waiting. Batch {bid} keeps running on Anthropic's side; "
                     f"run the same score command again to collect its results.")
            raise SystemExit(130)
        progress(f"[judge] batch {bid} ended: {c.succeeded} succeeded, {c.errored} errored, "
                 f"{c.expired} expired, {c.canceled} canceled; downloading results")

        failed = {}
        for res in self.client.messages.batches.results(bid):
            key = res.custom_id
            if key not in todo:
                continue  # response no longer in the scored set; nothing to attach the verdict to
            it, t = todo[key]
            rec = self._new_record(it, t)
            r = res.result
            try:
                if r.type == "succeeded":
                    _check_refusal(r.message)
                    raw = next(blk.text for blk in r.message.content if blk.type == "text")
                    rec.update(_interpret(it, RubricVerdict.model_validate_json(raw)))
                    rec["batch_id"] = bid
                elif r.type == "errored":
                    err = getattr(r.error, "error", r.error)
                    raise RuntimeError(f"batch request errored: {getattr(err, 'type', '')}: "
                                       f"{getattr(err, 'message', err)}")
                else:  # expired or canceled
                    raise RuntimeError(f"batch request {r.type}")
            except (RuntimeError, ValueError, StopIteration) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            self._store(rec)
            if rec["error"]:
                failed[key] = rec
        return failed


BATCH_POLL_SECONDS = 60
VERDICT_SCHEMA = anthropic.transform_schema(RubricVerdict)


def _check_refusal(msg):
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"judge refusal ({getattr(msg.stop_details, 'category', None)})")


def _interpret(item, verdict) -> dict:
    by_index = {p.index: p for p in verdict.points}
    if sorted(by_index) != list(range(1, len(item.key_points) + 1)):
        raise ValueError(f"judge returned indices {sorted(by_index)} for {len(item.key_points)} key points")
    present = [by_index[i].present for i in range(1, len(item.key_points) + 1)]
    return {"present": present, "rationales": [by_index[i].rationale for i in sorted(by_index)]}
