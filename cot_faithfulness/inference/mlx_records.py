"""File names and record contents for scripts/run_mlx.py (no MLX import, so tests can use it).

Greedy runs keep the original layout, one file per prompt:
    <example_id>__<variant_id>.json
Sampled runs write one file per sample k = 0, 1, ...:
    <example_id>__<variant_id>__s<k>.json
"""

from __future__ import annotations

from pathlib import Path

from cot_faithfulness.analysis.influence import map_to_original_letter, parse_answer
from cot_faithfulness.analysis.thinking import parse_final_answer, split_thinking


def record_path(out_dir: Path, example_id: str, variant_id: str, sample_index: int | None = None) -> Path:
    """sample_index=None gives the greedy name; an integer gives the sampled name."""
    if sample_index is None:
        return out_dir / f"{example_id}__{variant_id}.json"
    return out_dir / f"{example_id}__{variant_id}__s{sample_index}.json"


def parse_record_name(name: str) -> tuple[str, str, int]:
    """(example_id, variant_id, sample_index) from a file name; greedy files are sample 0."""
    parts = name.removesuffix(".json").split("__")  # example ids use single underscores only
    if len(parts) == 2:
        return parts[0], parts[1], 0
    if len(parts) == 3 and parts[2].startswith("s") and parts[2][1:].isdigit():
        return parts[0], parts[1], int(parts[2][1:])
    raise ValueError(f"not a generation file name: {name}")


def build_record(
    *,
    model_id: str,
    example,
    variant,
    cot: str,
    n_new_tokens: int,
    finish_reason: str | None,
    prompt_opens_think: bool = False,
    extra: dict | None = None,
) -> dict:
    """One generation record.

    The first block of fields is exactly what the November 2026 greedy runs wrote,
    in the same order. Thinking fields are added only when the trace has a thinking
    block, and `extra` (sample index and decoding settings) only for non-default runs,
    so a default greedy run still writes byte-for-byte the same JSON.
    """
    letters = list(variant.display_options.keys())
    parts = split_thinking(cot, prompt_opens_think)
    if parts["has_thinking"]:
        display_answer, answer_source = parse_final_answer(cot, letters, prompt_opens_think, parser=parse_answer)
    else:
        display_answer, answer_source = parse_answer(cot, letters), "whole_text"
    record = {
        "model_id": model_id,
        "backend": "mlx",
        "task": example.task,
        "example_id": example.example_id,
        "variant_id": variant.variant_id,
        "hint_type": variant.hint_type,
        "prompt": variant.prompt,
        "cot": cot,  # full output, thinking included
        "n_new_tokens": n_new_tokens,
        "finish_reason": finish_reason,  # "length" means it hit max_tokens
        "parsed_answer": display_answer,
        "original_answer": map_to_original_letter(display_answer, variant.letter_map),
        "correct_letter": example.correct_letter,
        "hint_suggested_letter": variant.hint_suggested_letter,
        "hint_description": variant.hint_description,
        "letter_map": variant.letter_map,
    }
    if parts["has_thinking"] or prompt_opens_think:
        record.update({
            "prompt_opens_thinking": prompt_opens_think,
            "thinking_closed": parts["thinking_closed"],
            "thinking": parts["thinking"],
            "answer_text": parts["answer_text"],
            "answer_source": answer_source,  # "answer_text", or "whole_text" when we fell back
        })
    if extra:
        record.update(extra)
    return record
