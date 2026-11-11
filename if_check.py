"""Programmatic constraint checker for the constrained instruction-following domain.

Usage:
    from ifcheck import score
    frac, detail = score(response_text, constraints_json)

`response_text` should be the extracted answer: the text after 'Final answer:' (Section III-D, stage 4).
Score = fraction of constraints satisfied, in [0, 1] (Section III-E, metric 1).
"""
import json
import re

WORD = re.compile(r"[A-Za-z0-9']+")

def _words(t):
    return WORD.findall(t)

def _lines(t):
    return [l for l in t.splitlines() if l.strip()]

def _has_word(t, w):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", t, re.I) is not None

def _count_word(t, w):
    return len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", t, re.I))

def _strip_json_fence(t):
    t = t.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    return m.group(1) if m else t

CHECKS = {
    "word_count": lambda t, p: p.get("min", 0) <= len(_words(t)) <= p.get("max", 10 ** 9),
    "bullet_count": lambda t, p: sum(1 for l in _lines(t) if re.match(r"\s*[-*] ", l)) == p["n"],
    "numbered_count": lambda t, p: sum(1 for l in _lines(t) if re.match(r"\s*\d+\. ", l)) == p["n"],
    "include_keywords": lambda t, p: all(_has_word(t, w) for w in p["words"]),
    "exclude_words": lambda t, p: not any(_has_word(t, w) for w in p["words"]),
    "keyword_frequency": lambda t, p: _count_word(t, p["word"]) >= p["min"],
    "no_commas": lambda t, p: "," not in t,
    "lowercase": lambda t, p: t == t.lower(),
    "uppercase": lambda t, p: t == t.upper(),
    "starts_with": lambda t, p: t.strip().lower().startswith(p["text"].lower()),
    "ends_with": lambda t, p: t.strip().endswith(p["text"]),
    "paragraph_count": lambda t, p: len([s for s in re.split(r"\n\s*\*\*\*\s*\n", t.strip()) if s.strip()]) == p["n"]
                                     and t.count("***") == p["n"] - 1,
    "json_keys": lambda t, p: _json_ok(t, p["keys"]),
    "title": lambda t, p: re.search(r"<<[^<>\n]+>>", t) is not None,
    "postscript": lambda t, p: re.search(r"(^|\n)\s*P\.S\.", t) is not None,
    "placeholder_count": lambda t, p: len(re.findall(r"\[[^\[\]\n]+\]", t)) >= p["min"],
    "line_count": lambda t, p: len(_lines(t)) == p["n"],
    "avoid_letter": lambda t, p: p["letter"].lower() not in t.lower(),
    "wrap_quotes": lambda t, p: len(t.strip()) >= 2 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "highlight_count": lambda t, p: len(re.findall(r"(?<!\*)\*[^\s*][^*\n]*\*(?!\*)", t)) >= p["min"],
    "section_count": lambda t, p: len(re.findall(r"(?m)^\s*SECTION \d+", t)) == p["n"],
    "capital_word_max": lambda t, p: sum(1 for w in _words(t) if len(w) > 1 and w.isupper()) <= p["max"],
    "each_line_max_words": lambda t, p: all(len(_words(l)) <= p["max"] for l in _lines(t)),
}

def _json_ok(t, keys):
    try:
        obj = json.loads(_strip_json_fence(t))
    except Exception:
        return False
    return isinstance(obj, dict) and set(obj.keys()) == set(keys)


def describe(c):
    """Natural-language wording of a constraint, used verbatim in the prompt."""
    k, p = c["type"], c
    if k == "word_count":
        if "min" in p and "max" in p:
            return f"Your response must contain between {p['min']} and {p['max']} words."
        if "max" in p:
            return f"Your response must contain no more than {p['max']} words."
        return f"Your response must contain at least {p['min']} words."
    return {
        "bullet_count": lambda: f"Use exactly {p.get('n')} bullet points, each on its own line starting with '- '.",
        "numbered_count": lambda: f"Format the content as a numbered list with exactly {p.get('n')} items, written '1. ', '2. ', and so on.",
        "include_keywords": lambda: "Include the words " + ", ".join(f"'{w}'" for w in p.get("words", [])) + ".",
        "exclude_words": lambda: "Do not use the words " + ", ".join(f"'{w}'" for w in p.get("words", [])) + " anywhere in your response.",
        "keyword_frequency": lambda: f"Use the word '{p.get('word')}' at least {p.get('min')} times.",
        "no_commas": lambda: "Do not use any commas in your response.",
        "lowercase": lambda: "Write your entire response in lowercase letters; no capital letters are allowed.",
        "uppercase": lambda: "Write your entire response in capital letters.",
        "starts_with": lambda: f"Begin your response with the exact words '{p.get('text')}'.",
        "ends_with": lambda: f"End your response with the exact phrase '{p.get('text')}', with nothing after it.",
        "paragraph_count": lambda: f"Write exactly {p.get('n')} paragraphs, separated from each other by a line containing only ***.",
        "json_keys": lambda: "Format your entire response as a single valid JSON object with exactly these keys: " + ", ".join(f'"{x}"' for x in p.get("keys", [])) + ". Output no text outside the JSON.",
        "title": lambda: "Include a title wrapped in double angular brackets, such as <<my title>>.",
        "postscript": lambda: "At the end of your response, add a postscript starting with 'P.S.'",
        "placeholder_count": lambda: f"Include at least {p.get('min')} placeholders in square brackets, such as [name].",
        "line_count": lambda: f"Your response must consist of exactly {p.get('n')} non-empty lines.",
        "avoid_letter": lambda: f"Do not use the letter '{p.get('letter')}' anywhere in your response, in either uppercase or lowercase.",
        "wrap_quotes": lambda: "Wrap your entire response in double quotation marks.",
        "highlight_count": lambda: f"Highlight at least {p.get('min')} phrases using markdown italics with single asterisks, like *this phrase*.",
        "section_count": lambda: f"Divide your response into exactly {p.get('n')} sections, each beginning with a line of the form 'SECTION 1', 'SECTION 2', and so on.",
        "capital_word_max": lambda: f"Use no more than {p.get('max')} words written entirely in capital letters.",
        "each_line_max_words": lambda: f"No line may contain more than {p.get('max')} words.",
    }[k]()


def score(response, constraints):
    if isinstance(constraints, str):
        constraints = json.loads(constraints)
    detail = [(c["type"], bool(CHECKS[c["type"]](response, c))) for c in constraints]
    return sum(ok for _, ok in detail) / len(detail), detail