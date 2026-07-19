#!/usr/bin/env python3
"""Re-parse answers from cached CoT without regenerating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.influence import parse_answer, map_to_original_letter
from cot_faithfulness.config import CACHE_DIR
from cot_faithfulness.inference.cache import iter_cached_records


def main() -> None:
    updated = 0
    for rec in iter_cached_records(CACHE_DIR):
        valid = list(rec.get("letter_map", {}).keys()) or ["A", "B", "C"]
        new_display = parse_answer(rec["cot"], valid)
        new_orig = map_to_original_letter(new_display, rec.get("letter_map", {}))
        if new_display != rec.get("parsed_answer") or new_orig != rec.get("original_answer"):
            rec["parsed_answer"] = new_display
            rec["original_answer"] = new_orig
            path = CACHE_DIR / "generations" / rec["model_id"].replace("/", "__") / (
                f"{rec['example_id']}__{rec['variant_id']}.json"
            )
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            updated += 1
    print(f"Updated {updated} cached records")


if __name__ == "__main__":
    main()
