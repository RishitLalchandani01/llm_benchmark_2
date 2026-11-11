"""Load the testing prompt matrix (Sec. III-C, Table II) and its ground-truth records."""
import csv
import json
import re
from dataclasses import dataclass, field
from typing import Optional

DOMAIN_CODES = {
    "Mathematical reasoning": "MATH",
    "Code generation": "CODE",
    "Factual and scientific knowledge": "FACT",
    "Scientific explanation": "EXPL",
    "Constrained instruction following": "INST",
}


@dataclass
class Item:
    item_id: str
    domain: str
    tier: str
    task_type: str
    answer_format: str
    prompt: str
    canonical_answer: str
    accepted_variants: list
    scoring_rule: str
    reference_answer: str
    key_points: list
    unit_tests: list
    entry_point: str
    constraints: Optional[list]
    tier_status: str
    expected_n_tests: Optional[int] = field(default=None)

    @property
    def rule(self) -> str:
        """Scorer name: numeric_match, normalized_exact_match, option_match, pass@1, rubric, programmatic."""
        return self.scoring_rule.split(":", 1)[0].strip()

    @property
    def domain_code(self) -> str:
        return DOMAIN_CODES[self.domain]


def split_tests(src: str) -> list:
    """Split unit_tests into assert blocks: a new block starts at every unindented line,
    except try/if continuations (except/else/elif/finally)."""
    blocks = []
    for line in src.splitlines():
        if not line.strip():
            continue
        continuation = line[0] in " \t" or re.match(r"(except|else|elif|finally)\b", line)
        if continuation and blocks:
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
    return blocks


def load_items(path: str) -> list:
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.search(r"(\d+) tests", r["scoring_rule"])
            items.append(Item(
                item_id=r["item_id"],
                domain=r["domain"],
                tier=r["tier"],
                task_type=r["task_type"],
                answer_format=r["answer_format"],
                prompt=r["prompt"],
                canonical_answer=r["canonical_answer"],
                accepted_variants=[v for v in r["accepted_variants"].split("|") if v],
                scoring_rule=r["scoring_rule"],
                reference_answer=r["reference_answer"],
                key_points=[k for k in r["key_points"].split("|") if k],
                unit_tests=split_tests(r["unit_tests"]),
                entry_point=r["entry_point"],
                constraints=json.loads(r["constraints_json"]) if r["constraints_json"] else None,
                tier_status=r["tier_status"],
                expected_n_tests=int(m.group(1)) if m else None,
            ))
    return items
