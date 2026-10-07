"""Global seeding and determinism controls (CLAUDE.md invariant 4).

CUDA nondeterminism note: some CUDA kernels (atomic reductions, certain cuDNN
convolutions) have no deterministic implementation. ``seed_everything`` enables
``torch.use_deterministic_algorithms`` in warn-only mode by default so runs are
reproducible where feasible without hard-crashing on an unavoidable nondeterministic
op. Set ``strict=True`` to make such ops raise instead. The hermetic CI path runs on
CPU, where determinism holds.

CUDA specifics handled here:
  - ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` is set for you (CuBLAS GEMMs are otherwise
    nondeterministic on CUDA >= 10.2). It must be set BEFORE the first CuBLAS call, so
    ``seed_everything`` is called at the very top of ``run_experiment``.
  - Memory-efficient/flash attention backward remains nondeterministic by default. We do
    NOT force it off, because the deterministic fallback is substantially slower; GPU
    runs are therefore reproducible up to attention-backward reduction order. Pass
    ``strict=True`` if a run must be bitwise reproducible, and expect a slowdown.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, strict: bool = False) -> int:
    """Seed python/numpy/torch RNGs and set deterministic flags.

    Returns the seed for convenience/logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CuBLAS GEMMs on CUDA >= 10.2; must precede the first
    # CuBLAS call, hence setting it here rather than relying on the shell.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    return seed


def torch_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded independently of global state.

    Prefer passing an explicit generator to samplers/dataloaders over mutating
    global RNG, so seeding is local and composable.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def numpy_rng(seed: int) -> np.random.Generator:
    """A local numpy ``Generator`` (used for Dirichlet partitioning, poisoning)."""
    return np.random.default_rng(seed)
