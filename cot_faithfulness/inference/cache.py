"""Disk cache for model generations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_path(cache_dir: Path, model_id: str, example_id: str, variant_id: str) -> Path:
    model_slug = model_id.replace("/", "__")
    return cache_dir / "generations" / model_slug / f"{example_id}__{variant_id}.json"


def load_cached(cache_dir: Path, model_id: str, example_id: str, variant_id: str) -> dict | None:
    path = cache_path(cache_dir, model_id, example_id, variant_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_cached(cache_dir: Path, record: dict[str, Any]) -> Path:
    path = cache_path(
        cache_dir,
        record["model_id"],
        record["example_id"],
        record["variant_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def iter_cached_records(cache_dir: Path, model_id: str | None = None) -> list[dict]:
    gen_dir = cache_dir / "generations"
    if not gen_dir.exists():
        return []

    records = []
    model_dirs = [gen_dir / model_id.replace("/", "__")] if model_id else gen_dir.iterdir()
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        for path in model_dir.glob("*.json"):
            with open(path) as f:
                records.append(json.load(f))
    return records
