"""Split reasoning-model output into its thinking block and its final answer.

Models trained to reason at length (Qwen3 with enable_thinking, DeepSeek-R1
distills, QwQ) write their reasoning inside <think> ... </think> and the answer
after it. Two shapes show up:

  1. The model writes both tags:          "<think>\\n...\\n</think>\\n\\nAnswer: (B)"
  2. The chat template already wrote
     "<think>" at the end of the prompt,
     so the output only has the close tag: "...\\n</think>\\n\\nAnswer: (B)"

We keep the whole output as the trace (the judge sees everything) and read the
answer from the text after the thinking block, falling back to the whole text
when that part has no answer.
"""

from __future__ import annotations

from typing import Callable

from cot_faithfulness.analysis.influence import parse_answer_v2

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def prompt_opens_thinking(prompt_text: str) -> bool:
    """True if the prompt ends inside an unclosed <think> block (shape 2 above).

    Qwen3 with enable_thinking=False writes an empty "<think>\\n\\n</think>" into the
    prompt; that block is closed, so this returns False.
    """
    return prompt_text.rfind(THINK_OPEN) > prompt_text.rfind(THINK_CLOSE)


def split_thinking(text: str, opened_in_prompt: bool = False) -> dict:
    """Return the thinking block and the answer text of one generation.

    has_thinking     the output has a thinking block (opened here or in the prompt)
    thinking_closed  the block was closed; False usually means the token limit hit
    thinking         text inside the block
    answer_text      text after the LAST </think> (Qwen3's chat template splits the
                     same way); empty when the block never closed
    """
    if THINK_CLOSE in text:
        before, _, after = text.rpartition(THINK_CLOSE)
        thinking = before.strip()
        if thinking.startswith(THINK_OPEN):
            thinking = thinking[len(THINK_OPEN):].strip()
        return {"has_thinking": True, "thinking_closed": True,
                "thinking": thinking, "answer_text": after.strip()}
    stripped = text.lstrip()
    if opened_in_prompt or stripped.startswith(THINK_OPEN):
        # Thinking started but never ended (usually cut off by max tokens).
        thinking = stripped[len(THINK_OPEN):] if stripped.startswith(THINK_OPEN) else stripped
        return {"has_thinking": True, "thinking_closed": False,
                "thinking": thinking.strip(), "answer_text": ""}
    return {"has_thinking": False, "thinking_closed": False, "thinking": "", "answer_text": text}


def parse_final_answer(
    text: str,
    valid_letters: list[str],
    opened_in_prompt: bool = False,
    parser: Callable[[str, list[str]], str | None] = parse_answer_v2,
) -> tuple[str | None, str]:
    """Answer letter and where it came from ("answer_text" or "whole_text").

    With no thinking block this is exactly parser(text, valid_letters), so
    ordinary instruct-model traces parse the same way as before.
    """
    parts = split_thinking(text, opened_in_prompt)
    if parts["has_thinking"] and parts["answer_text"]:
        letter = parser(parts["answer_text"], valid_letters)
        if letter is not None:
            return letter, "answer_text"
    return parser(text, valid_letters), "whole_text"
