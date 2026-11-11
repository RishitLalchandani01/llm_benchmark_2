"""Stage 2: inference backends.

Every backend takes chat messages plus decoding parameters and returns a Completion. Token counts
come from the serving engine, i.e. from each model's own tokenizer (Sec. III-D, logging).

  openai_compat  any OpenAI-compatible /v1/chat/completions server: vLLM, Ollama, LM Studio,
                 llama.cpp server, OpenRouter, Together, Fireworks, ...
  mlx            local Apple-silicon inference through mlx-lm (optional dependency)
  oracle         returns the ground-truth answer in the required format; for dry runs of the pipeline
"""
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Completion:
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_s: float
    finish_reason: Optional[str]
    served_model: Optional[str]


class OpenAICompatBackend:
    def __init__(self, base_url, api_key_env=None, timeout=600, max_retries=3, **_):
        from openai import OpenAI
        api_key = os.environ.get(api_key_env, "") if api_key_env else "not-needed"
        if api_key_env and not api_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    def generate(self, model, messages, temperature, top_p, max_tokens, seed, extra_body=None):
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, seed=seed, extra_body=extra_body or None,
        )
        dt = time.perf_counter() - t0
        choice = r.choices[0]
        return Completion(
            text=choice.message.content or "",
            input_tokens=r.usage.prompt_tokens if r.usage else None,
            output_tokens=r.usage.completion_tokens if r.usage else None,
            latency_s=dt,
            finish_reason=choice.finish_reason,
            served_model=r.model,
        )


class MLXBackend:
    """Local inference with mlx-lm (pip install mlx-lm). Loads one model at a time."""

    def __init__(self, **_):
        self._loaded = None

    def _load(self, model):
        if self._loaded is None or self._loaded[0] != model:
            from mlx_lm import load
            self._loaded = (model, *load(model))
        return self._loaded[1], self._loaded[2]

    def generate(self, model, messages, temperature, top_p, max_tokens, seed, extra_body=None):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        mdl, tok = self._load(model)
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0)
        t0 = time.perf_counter()
        text = generate(mdl, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
        dt = time.perf_counter() - t0
        n_out = len(tok.encode(text, add_special_tokens=False))
        return Completion(text, len(tok.encode(prompt, add_special_tokens=False)), n_out, dt,
                          "length" if n_out >= max_tokens else "stop", model)


class OracleBackend:
    """Answers every item correctly in the required format. Used to test the pipeline end to end:
    every deterministic scorer should give 1.0 (INST and EXPL have no canonical answer and score 0)."""

    def __init__(self, items=None, **_):
        self.by_prompt = {it.prompt: it for it in (items or [])}

    def generate(self, model, messages, temperature, top_p, max_tokens, seed, extra_body=None):
        user = messages[-1]["content"]
        item = next((it for p, it in self.by_prompt.items() if p in user), None)
        if item is None:
            text = "Final answer: unknown"
        elif item.rule == "pass@1":
            text = f"Final answer:\n```python\n{item.canonical_answer}\n```"
        elif item.rule == "rubric":
            text = f"Final answer: {item.reference_answer}"
        else:
            text = f"Final answer: {item.canonical_answer}"
        return Completion(text, len(user.split()), len(text.split()), 0.0, "stop", "oracle")


def make_backend(spec: dict, items=None):
    kind = spec["type"]
    if kind == "openai_compat":
        return OpenAICompatBackend(**spec)
    if kind == "mlx":
        return MLXBackend(**spec)
    if kind == "oracle":
        return OracleBackend(items=items)
    raise ValueError(f"unknown backend type: {kind}")
