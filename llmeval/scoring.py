"""Stage 5: score each response against its ground-truth record (Sec. III-E, metric 1).

Item score s in [0, 1]: binary for numeric, exact-match and option items, the unit-test pass rate for
code, and the fraction of satisfied constraints (instruction following) or rubric key points
(explanation, filled in by judge.py). 'correct' is s == 1, the binary outcome used by McNemar's test.
"""
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import unicodedata

from .extract import extract_code, first_line, split_marker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_check import score as ifcheck_score  # noqa: E402

WORD_LIMIT = 120
TEST_TIMEOUT_S = 10


# ---------- numeric_match ----------

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
FRAC = re.compile(r"\\[dt]?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}|(?<![\w.])(-?\d+)\s*/\s*(\d+)(?![\w.])")


def parse_number(s: str):
    """First number in s, after stripping $, %, thousands separators and units."""
    s = s.replace("\u2212", "-").replace("$", " ").replace("%", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    num, frac = NUM.search(s), FRAC.search(s)
    if frac and (num is None or frac.start() <= num.start()):
        a, b = (frac.group(1), frac.group(2)) if frac.group(1) else (frac.group(3), frac.group(4))
        return float(a) / float(b) if float(b) else None
    return float(num.group(0)) if num else None


def numeric_match(answer: str, canonical: str):
    x, y = parse_number(answer), float(canonical)
    ok = x is not None and abs(x - y) <= (1e-4 * abs(y) if y else 1e-9)
    return float(ok), {"parsed": x}


# ---------- normalized_exact_match / option_match ----------

ARTICLES = re.compile(r"^(?:the|a|an)\s+")


def normalize(s: str) -> str:
    """Lowercase, strip accents, punctuation, surrounding whitespace and leading articles."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P") and ch not in string.punctuation)
    s = " ".join(s.split())
    return ARTICLES.sub("", s).strip()


def exact_match(answer: str, canonical: str, variants: list):
    a = normalize(first_line(answer))
    ok = a in {normalize(v) for v in [canonical, *variants]}
    return float(ok), {"normalized": a}


def option_match(answer: str, canonical: str):
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", answer)
    got = m.group(1) if m else None
    return float(got == canonical.strip().upper()), {"option": got}


# ---------- pass@1 ----------

HARNESS = r'''
import json, signal, sys
T = float(sys.argv[1])
code = open("solution.py").read()
tests = json.load(open("tests.json"))
res = {"load_error": None, "tests": []}
class _Timeout(BaseException):
    pass
def _alarm(signum, frame):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
ns = {"__name__": "__solution__"}
try:
    signal.setitimer(signal.ITIMER_REAL, T)
    exec(compile(code, "solution.py", "exec"), ns)
except BaseException as e:
    res["load_error"] = repr(e)[:300]
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
if res["load_error"] is None:
    for t in tests:
        try:
            signal.setitimer(signal.ITIMER_REAL, T)
            exec(compile(t, "test", "exec"), ns)
            res["tests"].append(True)
        except BaseException:
            res["tests"].append(False)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
with open("result.json", "w") as f:
    json.dump(res, f)
'''


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))


def run_tests(code: str, tests: list, timeout_s: float = TEST_TIMEOUT_S):
    """Run the code, then each hidden test in the same namespace, in a separate isolated Python
    process (python -I, empty environment, temp working directory). This is process isolation only,
    not a security sandbox: run untrusted model code inside a container or VM."""
    with tempfile.TemporaryDirectory(prefix="llmeval_") as d:
        open(os.path.join(d, "solution.py"), "w").write(code)
        json.dump(tests, open(os.path.join(d, "tests.json"), "w"))
        open(os.path.join(d, "harness.py"), "w").write(HARNESS)
        try:
            subprocess.run([sys.executable, "-I", "harness.py", str(timeout_s)], cwd=d,
                           env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout_s * (len(tests) + 1) + 5, preexec_fn=_limits,
                           start_new_session=True)
        except subprocess.TimeoutExpired:
            return [False] * len(tests), "harness timeout"
        try:
            res = json.load(open(os.path.join(d, "result.json")))
        except (OSError, ValueError):
            return [False] * len(tests), "no result (process exited early)"
    passed = res["tests"] or [False] * len(tests)
    return passed, res["load_error"]


def pass_rate(code, item):
    if code is None:
        return 0.0, {"tests": [], "error": "no code extracted"}
    passed, err = run_tests(code, item.unit_tests)
    return sum(passed) / len(item.unit_tests), {"tests": passed, "error": err}


# ---------- programmatic (instruction following) ----------

def constraints_score(answer: str, item):
    frac, detail = ifcheck_score(answer, item.constraints)
    return frac, {"constraints": detail}


# ---------- dispatch ----------

def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s or ""))


def _score_text(item, answer):
    if answer is None:
        return 0.0, {}
    rule = item.rule
    if rule == "numeric_match":
        return numeric_match(answer, item.canonical_answer)
    if rule == "normalized_exact_match":
        return exact_match(answer, item.canonical_answer, item.accepted_variants)
    if rule == "option_match":
        return option_match(answer, item.canonical_answer)
    if rule == "programmatic":
        return constraints_score(answer, item)
    raise ValueError(rule)


def score_response(item, text: str) -> dict:
    """Deterministic scoring of one response. Rubric items come back with score None; judge.py fills
    them in from the 'judge_text' field."""
    tail, found = split_marker(text)
    row = {"rule": item.rule, "format_failure": not found}
    if item.rule == "pass@1":
        strict, lenient = extract_code(text, item.entry_point)
        row["format_failure"] = strict is None
        row["score"], d = pass_rate(strict, item)
        row["score_lenient"], _ = pass_rate(lenient, item) if lenient != strict else (row["score"], None)
        row["extracted"], row["detail"] = strict, d
    elif item.rule == "rubric":
        judged = tail if found else (text or "").strip()
        row.update(score=None, score_lenient=None, extracted=tail, judge_text=judged,
                   word_count=word_count(judged), over_word_limit=word_count(judged) > WORD_LIMIT, detail={})
    else:
        row["score"], row["detail"] = _score_text(item, tail)
        row["score_lenient"], _ = (row["score"], None) if found else _score_text(item, (text or "").strip())
        row["extracted"] = tail
    return row
