"""Reproducibility utilities for deterministic experiments.

Provides comprehensive seed control across Python, NumPy, PyTorch,
and OS-level random number generators to ensure experiment reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for full reproducibility.

    Controls randomness across Python stdlib, NumPy, PyTorch CPU/GPU,
    and CUDA operations. Sets environment variables for hash randomization.

    Args:
        seed: The integer seed value to use across all RNGs.
        deterministic: If True, force PyTorch to use deterministic algorithms
            (may reduce performance but guarantees exact reproducibility).

    Example:
        >>> set_seed(42)
        >>> import numpy as np
        >>> np.random.rand()  # deterministic output
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # Older PyTorch versions

    log.info(f"Global random seed set to {seed} (deterministic={deterministic})")


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a seeded NumPy random number generator.

    Preferred over legacy np.random for new code — produces
    independent, reproducible streams without global state mutation.

    Args:
        seed: Seed for the RNG. If None, system entropy is used.

    Returns:
        A numpy Generator object for sampling operations.

    Example:
        >>> rng = get_rng(42)
        >>> rng.integers(0, 100, size=5)
    """
    return np.random.default_rng(seed)
