"""Experiment configuration for CoT faithfulness replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# BBH subtasks with clear multi-step reasoning chains
BBH_TASKS = [
    "logical_deduction_three_objects",
    "causal_judgement",
    "formal_fallacies",
    "temporal_sequences",
]

# Model family for size comparison (4-bit on CUDA T4)
MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Placebo added per protocol (prompt perturbation control)
HINT_TYPES = ["sycophancy", "reorder", "placebo"]

JUDGE_MODEL = "gpt-4o"
HUMAN_LABEL_TARGET = 30
MIN_KAPPA = 0.6
MIN_INFLUENCED_FOR_CI = 20
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CI = 0.95

MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0

# Preregistered item budget (Sec. power analysis)
DEFAULT_MAX_SAMPLES_PER_TASK = 250


@dataclass
class ExperimentConfig:
    """Runtime experiment settings."""

    tasks: list[str] = field(default_factory=lambda: list(BBH_TASKS))
    models: list[str] = field(default_factory=lambda: list(MODEL_CANDIDATES))
    max_samples_per_task: int | None = DEFAULT_MAX_SAMPLES_PER_TASK
    hint_types: list[str] = field(default_factory=lambda: list(HINT_TYPES))
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    seed: int = 42
    force_regenerate: bool = False
    probe_only: bool = False

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
