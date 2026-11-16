#!/usr/bin/env python3
"""Draw the 200-item subset for the preregistered stochastic sub-study.

The July 2026 plan says: "A secondary run at T = 0.7 with n = 5 samples per prompt
on a 200-item subset". It does not say how the 200 are picked. We fix that here,
before any sampled generation exists:

  * 50 items from each of the four BBH tasks (so every task counts equally),
  * drawn uniformly at random, without looking at any model output,
  * with Python's random.Random seeded by "<SEED>:<task>" (SEED = 20261110,
    the date this rule was written), so each task's draw does not depend on
    the order the tasks are listed in.

    python scripts/make_substudy_subset.py      # -> data/substudy_subset_200.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.config import BBH_TASKS, PROJECT_ROOT

SEED = 20261110
PER_TASK = 50
DEFAULT_OUT = PROJECT_ROOT / "data" / "substudy_subset_200.json"


def item_number(example_id: str) -> int:
    """'causal_judgement_17' -> 17, so ids sort as 0, 1, 2, ... and not as text."""
    return int(example_id.rsplit("_", 1)[1])


def draw_subset(ids_by_task: dict[str, list[str]], per_task: int = PER_TASK, seed: int = SEED) -> list[str]:
    """per_task ids from every task; same inputs and seed always give the same list."""
    picked = []
    for task in sorted(ids_by_task):
        ids = sorted(ids_by_task[task], key=item_number)
        if len(ids) < per_task:
            raise ValueError(f"{task} has only {len(ids)} items")
        rng = random.Random(f"{seed}:{task}")
        picked += sorted(rng.sample(ids, per_task), key=item_number)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from cot_faithfulness.data.bbh import load_all_tasks  # needs the BBH download, so imported here

    ids_by_task = defaultdict(list)
    for ex in load_all_tasks(BBH_TASKS, 250):  # the same 937 items as the greedy run
        ids_by_task[ex.task].append(ex.example_id)
    subset = draw_subset(ids_by_task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(subset, open(args.out, "w"), indent=1)
    print(f"{len(subset)} items ({PER_TASK} per task, seed {SEED}) -> {args.out}")


if __name__ == "__main__":
    main()
