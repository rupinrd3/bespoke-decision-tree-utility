#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parallel execution helpers used across the Bespoke Decision Tree utility.

This module centralises small utilities for deciding when parallel execution
is worthwhile and for chunking work items. Keeping the logic here avoids
duplicating psutil checks and chunk-size heuristics in multiple call sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import psutil


@dataclass(frozen=True)
class ParallelPolicy:
    """Lightweight container for parallel execution thresholds."""

    min_samples: int = 1000
    memory_threshold_gb: float = 8.0
    fallback: str = "sequential"


def should_parallelize(
    work_items: int,
    policy: ParallelPolicy,
    extra_memory_overhead_gb: float = 0.0,
) -> bool:
    """
    Determine whether it is worthwhile to dispatch work to multiple processes.

    Args:
        work_items: Total number of work units (rows, features, etc.).
        policy: Threshold configuration.
        extra_memory_overhead_gb: Caller-estimated memory footprint per worker.

    Returns:
        True if parallel execution is recommended, False otherwise.
    """
    if work_items < policy.min_samples:
        return False

    try:
        sys_memory = psutil.virtual_memory()
    except Exception:
        return False

    available_gb = sys_memory.available / (1024**3)
    effective_available = available_gb - extra_memory_overhead_gb

    if effective_available <= 0:
        return False

    if sys_memory.percent >= 80.0:
        return False

    return effective_available >= policy.memory_threshold_gb


def normalise_n_jobs(n_jobs: int | None) -> int:
    """
    Convert Joblib-style n_jobs hints into an explicit worker count.

    Joblib treats negative values as "CPU count + 1 + n_jobs". We mirror that
    behaviour so we can reason about chunk sizes ahead of time.
    """
    cpu_total = psutil.cpu_count(logical=True) or 1

    if n_jobs is None or n_jobs == 0:
        return 1

    if n_jobs < 0:
        requested = cpu_total + 1 + n_jobs
        return max(1, requested)

    return min(cpu_total, n_jobs)


def iter_chunks(
    total_items: int,
    *,
    min_chunk_size: int = 500,
    max_chunk_size: int = 5000,
    n_jobs: int = 1,
) -> Iterator[Tuple[int, int]]:
    """
    Yield `(start, end)` tuples describing work chunks.

    Args:
        total_items: Total number of items to process.
        min_chunk_size: Lower bound for chunk size.
        max_chunk_size: Upper bound for chunk size.
        n_jobs: Intended parallel worker count.
    """
    if total_items <= 0:
        return

    if n_jobs <= 0:
        n_jobs = 1

    raw_size = max(1, total_items // (n_jobs * 4))
    chunk_size = int(_clamp(raw_size, min_chunk_size, max_chunk_size))

    for start in range(0, total_items, chunk_size):
        end = min(total_items, start + chunk_size)
        yield start, end


def _clamp(value: int | float, minimum: int | float, maximum: int | float) -> int | float:
    return max(minimum, min(maximum, value))


def estimate_memory_overhead(
    batch_size: int,
    dtype_size_bytes: int,
    copies_per_worker: int,
    *,
    safety_multiplier: float = 1.5,
) -> float:
    """
    Rough memory overhead estimation (in GB) used for gating parallel work.
    """
    bytes_required = batch_size * dtype_size_bytes * copies_per_worker * safety_multiplier
    return bytes_required / (1024**3)


def slice_records(matrix: Sequence, start: int, end: int):
    """
    Thin wrapper around slicing that tolerates both numpy arrays and pandas values.
    """
    if hasattr(matrix, "iloc"):
        return matrix.iloc[start:end]
    return matrix[start:end]

