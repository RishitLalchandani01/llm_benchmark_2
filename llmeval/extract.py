"""Stage 4: answer extraction (Sec. III-D).

The text after the last 'Final answer:' marker is the extracted answer. A response without the marker
is a format failure: strict scoring gives it 0. Lenient extraction (used only for the secondary
'score_lenient' column and for judging) falls back to the whole response so format failures can be
separated from wrong answers.
"""
import re

# Tolerates markdown emphasis around the marker: **Final answer:**, **Final Answer**:, ...
MARKER = re.compile(r"[*_#\s]*final\s+answer\s*[*_]*\s*:\s*[*_]*[ \t]*", re.I)
FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def split_marker(text: str):
    """(text after the last marker, marker_found)."""
    matches = list(MARKER.finditer(text or ""))
    if not matches:
        return None, False
    return text[matches[-1].end():].strip(), True


def first_line(s: str) -> str:
    for line in s.splitlines():
        if line.strip():
            return line.strip()
    return ""


def extract_code(text: str, entry_point: str):
    """(code_strict, code_lenient). Strict: a fenced block after the marker, or the raw tail if it
    defines a function. Lenient: otherwise the last fenced block anywhere that defines entry_point."""
    tail, found = split_marker(text)
    strict = None
    if found:
        blocks = FENCE.findall(tail)
        if blocks:
            strict = blocks[0]
        elif re.search(r"^\s*def\s+\w+\s*\(", tail, re.M):
            strict = tail
    lenient = strict
    if lenient is None:
        blocks = FENCE.findall(text or "")
        defining = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
        if defining or blocks:
            lenient = (defining or blocks)[-1]
    return strict, lenient
