"""Fixed-prompt LLM judge for the scientific-explanation rubric (Sec. III-E, metric 1).

The judge is a Claude model, outside every evaluated family, which avoids self-preference bias.
It sees the question, the key points and the response, never the model's identity, and marks each key
point present or absent. Item score = fraction of key points present. Verdicts are cached in
<run_dir>/judge_cache.jsonl keyed by (judge model, prompt version, item, response text), so re-scoring
costs nothing. The judge model is held fixed: no automatic fallback to another model on a refusal; a
refused or failed item is left unscored and flagged for human grading instead.
"""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel

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
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"judge refusal ({getattr(resp.stop_details, 'category', None)})")
        verdict = resp.parsed_output
        by_index = {p.index: p for p in verdict.points}
        if sorted(by_index) != list(range(1, len(item.key_points) + 1)):
            raise ValueError(f"judge returned indices {sorted(by_index)} for {len(item.key_points)} key points")
        present = [by_index[i].present for i in range(1, len(item.key_points) + 1)]
        return {"present": present, "rationales": [by_index[i].rationale for i in sorted(by_index)],
                "request_id": resp._request_id}

    def grade(self, item, text) -> dict:
        key = cache_key(self.model, item.item_id, text)
        if key in self.cache:
            return self.cache[key]
        rec = {"key": key, "item_id": item.item_id, "judge_model": self.model,
               "prompt_version": JUDGE_PROMPT_VERSION, "error": None}
        if not text.strip():
            rec.update(present=[False] * len(item.key_points), rationales=["empty response"] * len(item.key_points))
        else:
            try:
                rec.update(self._call(item, text))
            except (anthropic.APIError, RuntimeError, ValueError) as e:
                rec["error"] = f"{type(e).__name__}: {e}"
        with self.lock:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["error"] is None:
                self.cache[key] = rec
        return rec

    def grade_many(self, pairs, progress=print):
        """pairs: list of (item, text). Returns verdicts in the same order."""
        todo = sum(cache_key(self.model, it.item_id, t) not in self.cache for it, t in pairs)
        progress(f"[judge] {len(pairs)} rubric responses, {todo} not yet cached ({self.model})")
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda p: self.grade(*p), pairs))
