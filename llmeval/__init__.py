"""Evaluation framework for 'Diminishing Returns in Large Language Models' (Section III).

Pipeline stages (Fig. 1, Sec. III-D):
  1. prompting.py   prompt construction (fixed zero-shot template, model's own chat format)
  2. backends.py    inference (OpenAI-compatible servers, local MLX, oracle for dry runs)
  3. runner.py      logging (raw text, token counts, latency, hardware, model version)
  4. extract.py     answer extraction after 'Final answer:'
  5. scoring.py     scoring against the ground-truth record; metrics.py / stats.py aggregate
"""
