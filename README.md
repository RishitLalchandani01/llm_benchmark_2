# llmeval: evaluation framework for "Diminishing Returns in Large Language Models"

An implementation of Section III of the paper: the testing prompt matrix (300 items, 5 domains × 3 tiers),
the five-stage execution pipeline (Fig. 1), the metrics in Eqs. 1–6, and the statistical analysis in Sec. III-F.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # rubric judge (Scientific explanation items)
export OPENROUTER_API_KEY=...       # only if a config uses an API backend
```

## Workflow

```bash
PY=.venv/bin/python
$PY -m llmeval --config config.local.yaml validate        # dataset + scorers vs ground truth, no models
$PY -m llmeval --config config.local.yaml run             # inference + logging (greedy + K=5 samples)
$PY -m llmeval --config config.local.yaml run --models qwen2.5-0.5b --items MATH-B,CODE-B --runs greedy   # smoke test
$PY -m llmeval --config config.local.yaml score           # extraction + scoring; EXPL items go to the Claude judge
$PY -m llmeval --config config.local.yaml similarity      # Eq. 2 + BERTScore (pip install sentence-transformers bert-score)
$PY -m llmeval --config config.local.yaml analyze         # metrics, fits, CIs, tests, figures, summary.md
$PY -m llmeval --config config.local.yaml human-export --n 100   # blinded sheets for two human graders
$PY -m llmeval --config config.local.yaml human-import    # Cohen's kappa: grader A vs B, judge vs each
$PY -m pytest tests
```

Every stage is resumable: `run` appends to `runs/<run_name>/responses/<model>.jsonl` and skips what is logged;
`score` caches code-execution results and judge verdicts, so re-running costs nothing.

## Configs

| file | what it runs |
|---|---|
| `config.local.yaml` | what fits on this Mac (M4 Pro, 24 GB) through Ollama: Qwen2.5 0.5–14B, Llama 3.1 8B, Phi-3-mini, Mistral 7B, Gemma 2 2B/9B, all at Q4_K_M |
| `config.paper.yaml` | the full Sec. III-B roster at bf16: vLLM on a GPU host, plus Llama 3.1 405B through a pinned API provider |
| `config.oracle.yaml` | a dry run with no models: the oracle backend answers with the ground truth |

Backends are pluggable: `openai_compat` (vLLM, Ollama, LM Studio, llama.cpp, OpenRouter, Together, …),
`mlx` (local Apple silicon via mlx-lm) and `oracle`. A model's backend, served name, parameter count,
precision and version string are set per model in the config, so one ladder can move from this Mac to a GPU
box by changing `backend:`. Keep one serving setup and one precision within a ladder, as Sec. III-A and III-B require.

## How each piece maps to the paper

| paper | code |
|---|---|
| III-C prompt matrix, ground-truth record | `llmeval/items.py`; `validate` checks cell balance and that every canonical answer scores 1.0 |
| III-D 1. prompt construction | `llmeval/prompting.py`: one system instruction + question + required `Final answer:` line, sent as chat messages so each engine applies the model's own chat template |
| III-D 2. inference | `llmeval/backends.py`, `llmeval/runner.py`: greedy run (T=0) + K=5 samples (T=0.7, top-p 0.95, seeds `seed+k`), 1024-token cap, seeded random item order |
| III-D 3. logging | raw text, input/output tokens from the engine (the model's own tokenizer), latency, hardware, model version, precision, seed |
| III-D 4. answer extraction | `llmeval/extract.py`: text after the last `Final answer:`; missing marker = format failure (score 0, counted separately) |
| III-D 5. scoring | `llmeval/scoring.py`: numeric match (rel. tol. 1e-4), normalized exact match, option match, pass@1 (each hidden test in the same namespace, 10 s timeout), `if_check.score` for constraints |
| III-E 1. accuracy (Eq. 1) | rubric items: `llmeval/judge.py` (Claude, fixed prompt, blinded) + `llmeval/human.py` |
| III-E 2. semantic divergence (Eq. 2) | `llmeval/similarity.py` (FACT: extracted vs canonical answer; EXPL: full response vs reference answer) |
| III-E 3–5. variance, TER, U (Eqs. 3–5) | `llmeval/analysis.py` |
| III-E 5. MU per doubling (Eq. 6), log-linear vs saturating fit, AIC, N90 | `llmeval/analysis.py` |
| III-F bootstrap CIs, McNemar + Holm, per-domain/tier | `llmeval/analysis.py`; tables and figures by `llmeval/report.py` |

## Decisions that are not spelled out in the paper

These are worth stating in the paper's method section.

- **Strict vs lenient scores.** `score` follows the paper: no `Final answer:` marker means 0.
  `score_lenient` falls back to the whole response (for code, the last fenced block that defines the entry point),
  so you can show how much of a small model's deficit is format failure rather than wrong answers.
- **Binary correctness for McNemar**: `correct = (item score == 1)`, so code needs every test to pass,
  instruction items need every constraint met, and rubric items need every key point.
- **Rubric input.** The judge grades the extracted answer, or the whole response when the marker is missing
  (the strict score is still 0 in that case). The 120-word flag uses the same text.
- **Judge.** Claude Opus 5.5 (`claude-opus-5-5`, effort `medium`), with a fixed prompt versioned as `rubric-v1`.
  It sees the question, the key points and the response, never the model's name. The current Claude API
  takes no `temperature` parameter, so judge determinism comes from the fixed prompt and the verdict cache,
  not from sampling settings. There is deliberately no automatic fallback to another model on a refusal,
  which would change the grader mid-study; refused items keep `judge_error` and need a human grade.
  Validate the judge with `human-export` / `human-import` and report the kappas.
- **TER and U** use accuracy in percentage points. U (Eq. 5) is in points per teraFLOP, with
  FLOPs per response = 2 · N_a · (mean input tokens + mean output tokens).
- **AIC** counts the error variance as a parameter (k = 3 log-linear, k = 4 saturating). AICc is also reported.
  A 3-model ladder (Llama 3.1) gives a 3-parameter saturating fit with zero residual degrees of freedom,
  so its AIC comparison is not informative: report MU and McNemar for that ladder and lean on the 7-point Qwen ladder.
  `frac_boot_favour_saturating` is the share of item-bootstrap replicates in which the saturating curve wins.
- **Comparators** are compared with the primary ladder's fitted curve at their parameter count (`comparators.csv`).
- **Abstract placeholders** [X], [Y], [Z] are written to `analysis/headline.json`; the "mid-sized" model is
  `analysis.mid_model` in the config.

## Safety

`pass@1` executes model-written code. The harness runs it in a separate `python -I` process with an empty
environment, a temporary working directory and per-test timeouts. That is process isolation, not a sandbox,
so run the `score` step for real experiments inside a container or VM.

## Not implemented

The pilot run that reassigns tiers (Sec. III-C, criterion 5; every item is still `provisional`) and the
guided-completion contamination probes (criterion 2) are dataset-construction steps, not part of this pipeline.
