"""Stage 1: prompt construction (Sec. III-D).

One fixed zero-shot template for every item and model: a single system instruction, the question,
and a required closing line 'Final answer: <answer>'. The template is sent as chat messages, so the
serving engine renders it with each model's own chat format.
"""

SYSTEM_INSTRUCTION = (
    "You are answering a single question. You may reason before answering. "
    "End your response with a line of the form 'Final answer: <answer>'. "
    "Only the text after 'Final answer:' is graded, so put your complete answer there: "
    "a single number for numeric questions, the short answer or option letter for knowledge questions, "
    "the complete Python function in a ```python code block for programming tasks, "
    "and the full written response for writing and explanation tasks, following every instruction in the question."
)

CLOSING_LINE = "End your response with the line: Final answer: <answer>"

TEMPLATE_VERSION = "v1"


def build_messages(item, system_in_user: bool = False) -> list:
    """Chat messages for one item. system_in_user folds the system instruction into the user turn,
    for chat formats without a system role (e.g. Gemma 2)."""
    user = f"{item.prompt}\n\n{CLOSING_LINE}"
    if system_in_user:
        return [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{user}"}]
    return [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": user}]
