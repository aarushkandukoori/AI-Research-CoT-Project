"""BBH dataset loading and MCQ parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

LETTER_RE = re.compile(r"^\(([A-Z])\)\s*(.+)$", re.MULTILINE)
DASH_OPTION_RE = re.compile(r"^-\s+(.+)$", re.MULTILINE)
ANSWER_LETTER_RE = re.compile(r"\(([A-Z])\)")


@dataclass
class MCQExample:
    """A BBH question reformatted as multiple-choice."""

    task: str
    example_id: str
    question_stem: str
    options: dict[str, str]  # letter -> text
    correct_letter: str
    raw_input: str
    raw_target: str

    @property
    def option_letters(self) -> list[str]:
        return sorted(self.options.keys())

    def format_options(self, letter_order: list[str] | None = None) -> str:
        order = letter_order or self.option_letters
        lines = []
        for letter in order:
            lines.append(f"({letter}) {self.options[letter]}")
        return "\n".join(lines)

    def pick_wrong_letter(self, avoid: str | None = None) -> str:
        wrong = [l for l in self.option_letters if l != self.correct_letter and l != avoid]
        if not wrong:
            raise ValueError(f"No wrong option for {self.example_id}")
        # Prefer second option in list as "plausible" wrong answer
        for letter in self.option_letters:
            if letter in wrong:
                return letter
        return wrong[0]


def _split_stem_and_options(input_text: str) -> tuple[str, list[tuple[str, str]]]:
    if "Options:" not in input_text:
        raise ValueError("Expected 'Options:' section in BBH input")

    stem, options_block = input_text.split("Options:", 1)
    stem = stem.strip()
    options_block = options_block.strip()

    letter_matches = list(LETTER_RE.finditer(options_block))
    if letter_matches:
        options = [(m.group(1), m.group(2).strip()) for m in letter_matches]
        return stem, options

    dash_lines = DASH_OPTION_RE.findall(options_block)
    if dash_lines:
        letters = [chr(ord("A") + i) for i in range(len(dash_lines))]
        options = list(zip(letters, [line.strip() for line in dash_lines]))
        return stem, options

    raise ValueError(f"Could not parse options from: {options_block[:200]}")


def _target_to_letter(target: str, options: dict[str, str]) -> str:
    target = target.strip()
    letter_match = ANSWER_LETTER_RE.search(target)
    if letter_match:
        return letter_match.group(1)

    target_norm = target.lower().strip()
    for letter, text in options.items():
        if text.lower().strip() == target_norm:
            return letter
        if target_norm in text.lower():
            return letter
    raise ValueError(f"Could not map target '{target}' to option letter")


def parse_bbh_example(task: str, idx: int, raw_input: str, raw_target: str) -> MCQExample:
    stem, option_pairs = _split_stem_and_options(raw_input)
    options = {letter: text for letter, text in option_pairs}
    correct_letter = _target_to_letter(raw_target, options)
    return MCQExample(
        task=task,
        example_id=f"{task}_{idx}",
        question_stem=stem,
        options=options,
        correct_letter=correct_letter,
        raw_input=raw_input,
        raw_target=raw_target,
    )


def load_bbh_mcq(task: str, max_samples: int | None = None) -> list[MCQExample]:
    dataset = load_dataset("lukaemon/bbh", task, split="test")
    examples = []
    for idx, row in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        try:
            examples.append(parse_bbh_example(task, idx, row["input"], row["target"]))
        except ValueError:
            continue
    return examples


def load_all_tasks(tasks: list[str], max_samples_per_task: int | None = None) -> list[MCQExample]:
    all_examples = []
    for task in tasks:
        all_examples.extend(load_bbh_mcq(task, max_samples_per_task))
    return all_examples
