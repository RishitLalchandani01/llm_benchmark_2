"""Sanity checks on the prompt matrix and the scorers, run against the ground truth itself:
cell balance (5 domains x 3 tiers), canonical answers parse and score 1.0, every canonical solution
passes its hidden tests, test counts match the scoring rule, and every constraint type is known."""
from collections import Counter

from .scoring import exact_match, numeric_match, option_match, run_tests
from if_check import CHECKS  # importable once .scoring has put the project root on sys.path


def validate(items) -> bool:
    problems = []
    cells = Counter((it.domain_code, it.tier) for it in items)
    if len(set(cells.values())) != 1:
        problems.append(f"unbalanced cells: {dict(cells)}")
    print(f"{len(items)} items in {len(cells)} cells of {sorted(set(cells.values()))} items")

    for it in items:
        r = it.rule
        if r == "numeric_match" and numeric_match(it.canonical_answer, it.canonical_answer)[0] != 1:
            problems.append(f"{it.item_id}: canonical answer does not parse as a number")
        elif r == "normalized_exact_match":
            for v in [it.canonical_answer, *it.accepted_variants]:
                if exact_match(v, it.canonical_answer, it.accepted_variants)[0] != 1:
                    problems.append(f"{it.item_id}: variant {v!r} does not match itself")
        elif r == "option_match" and option_match(it.canonical_answer, it.canonical_answer)[0] != 1:
            problems.append(f"{it.item_id}: option answer {it.canonical_answer!r} is not A-D")
        elif r == "pass@1":
            if it.expected_n_tests != len(it.unit_tests):
                problems.append(f"{it.item_id}: {len(it.unit_tests)} test blocks, rule says {it.expected_n_tests}")
            passed, err = run_tests(it.canonical_answer, it.unit_tests)
            if not all(passed):
                problems.append(f"{it.item_id}: canonical solution fails tests {passed} ({err})")
        elif r == "programmatic":
            unknown = [c["type"] for c in it.constraints if c["type"] not in CHECKS]
            if unknown:
                problems.append(f"{it.item_id}: unknown constraint types {unknown}")
        elif r == "rubric" and not (it.key_points and it.reference_answer):
            problems.append(f"{it.item_id}: rubric item without key points or reference answer")

    for p in problems:
        print("FAIL", p)
    print("all checks passed" if not problems else f"{len(problems)} problems")
    return not problems
