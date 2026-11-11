import numpy as np
import pandas as pd

from llmeval.analysis import analyze, fit_curves, holm, mcnemar, n90, saturating
from llmeval.config import ModelSpec
from llmeval.extract import extract_code, split_marker
from llmeval.items import split_tests
from llmeval.scoring import exact_match, numeric_match, option_match, parse_number, run_tests


def test_marker_variants():
    assert split_marker("work...\nFinal answer: 29") == ("29", True)
    assert split_marker("**Final Answer:** 29") == ("29", True)
    assert split_marker("**Final answer**: $29") == ("$29", True)
    assert split_marker("Final answer: 1\nwait, Final answer: 2")[0] == "2"
    assert split_marker("The answer is 29.") == (None, False)


def test_numeric():
    assert parse_number("$1,260.00 dollars") == 1260
    assert parse_number(r"\boxed{11.25}") == 11.25
    assert parse_number("3/4 of a cup") == 0.75
    assert parse_number("−3") == -3
    assert numeric_match("87.48", "87.48")[0] == 1
    assert numeric_match("87.5", "87.48")[0] == 0
    assert numeric_match("about 29 dollars", "29")[0] == 1


def test_exact_and_option():
    assert exact_match("The Indian Ocean.", "Indian Ocean", [])[0] == 1
    assert exact_match("CO₂", "CO2", [])[0] == 1
    assert exact_match("Nitrogen", "N", ["nitrogen"])[0] == 1
    assert exact_match("Oxygen", "N", ["nitrogen"])[0] == 0
    assert option_match("(A) Dolphin", "A")[0] == 1
    assert option_match("B", "A")[0] == 0


def test_code_extraction_and_tests():
    text = "Here:\nFinal answer:\n```python\ndef f(x):\n    return x + 1\n```"
    strict, lenient = extract_code(text, "f")
    assert strict.strip().startswith("def f")
    passed, err = run_tests(strict, ["assert f(1) == 2", "assert f(1) == 3"])
    assert passed == [True, False] and err is None
    # code before the marker: format failure under strict, recovered by lenient
    strict, lenient = extract_code("```python\ndef f(x):\n    return x\n```\nFinal answer: see above", "f")
    assert strict is None and "def f" in lenient


def test_test_timeout_and_crash():
    passed, _ = run_tests("def f():\n    while True: pass", ["assert f() is None", "assert 1 == 1"], timeout_s=1)
    assert passed == [False, True]
    passed, err = run_tests("import os\nos._exit(0)", ["assert True"])
    assert passed == [False] and err


def test_split_tests_try_block():
    src = "assert f(0) == 1\ntry:\n    f(-1)\n    assert False\nexcept ValueError:\n    pass\nassert f(1) == 1"
    assert len(split_tests(src)) == 3


def test_stats():
    b, c, p = mcnemar([1] * 20 + [0] * 80, [0] * 20 + [0] * 80)
    assert (b, c) == (20, 0) and p < 1e-5
    assert list(holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]
    assert n90([1, 2, 4, 8], [0.2, 0.5, 0.85, 0.9]) == 4


def test_fit_prefers_saturating_on_saturating_data():
    N = np.array([0.5, 1.5, 3, 7, 14, 32, 72])
    A = saturating(N, 0.85, 0.35, 0.6)
    f = fit_curves(N, A)
    assert f["delta_aic"] > 0 and abs(f["sat_A_max"] - 0.85) < 0.02


def test_analyze_synthetic_ladder():
    rng = np.random.default_rng(0)
    N = [0.5, 1.5, 3, 7, 14, 32, 72]
    models = [ModelSpec(id=f"m{n}", backend="x", model="x", params_b=n, ladder="primary") for n in N]
    p_item = rng.uniform(0, 1, 300)  # item difficulty; a model solves items easier than its skill
    rows = []
    for m in models:
        skill = saturating(m.params_b, 0.85, 0.35, 0.6)
        for q in range(300):
            for run in ["greedy", "sample1", "sample2"]:
                ok = float(p_item[q] < skill + (0 if run == "greedy" else rng.normal(0, 0.05)))
                rows.append({"model_id": m.id, "item_id": f"Q{q:03d}", "run": run, "domain": "MATH",
                             "tier": "basic", "score": ok, "correct": ok, "format_failure": False,
                             "input_tokens": 100, "output_tokens": 200})
    res = analyze(pd.DataFrame(rows), models, n_boot=500, fit_boot=50)
    mu = res["marginal_utility"]
    mu = mu[mu.subset == "all"].MU_points_per_doubling.to_numpy()
    assert mu[0] > mu[-1]  # diminishing returns recovered
    fit = res["curve_fits"].query("subset == 'all'").iloc[0]
    assert fit.delta_aic > 0
    assert set(res["mcnemar"].columns) >= {"p", "p_holm"}


def test_judge_with_stub_client(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from llmeval import judge as J
    from llmeval.items import load_items

    item = next(it for it in load_items("llm_eval_prompt_matrix_300.csv") if it.rule == "rubric")
    n = len(item.key_points)
    calls = []

    class Stub:
        def parse(self, **kw):
            calls.append(kw)
            pts = [J.PointJudgement(index=i, rationale="r", present=i <= 2) for i in range(1, n + 1)]
            return SimpleNamespace(stop_reason="end_turn", parsed_output=J.RubricVerdict(points=pts), _request_id="req_1")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    j = J.Judge({"model": "claude-opus-5-5", "effort": "medium", "max_tokens": 16000, "concurrency": 2},
                str(tmp_path / "cache.jsonl"))
    j.client = SimpleNamespace(messages=Stub())
    v = j.grade(item, "some explanation")
    assert v["present"] == [True, True] + [False] * (n - 2)
    assert calls[0]["model"] == "claude-opus-5-5" and calls[0]["output_format"] is J.RubricVerdict
    assert "temperature" not in calls[0]
    j.grade(item, "some explanation")
    assert len(calls) == 1  # second call served from the cache
