import os
import random
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Choose the best available device following CUDA > MPS > CPU."""
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_project_root_on_path(project_root: str) -> None:
    """Add project root to sys.path if missing."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def create_timestamped_run_dir(base_dir: str) -> str:
    """Create base_dir (if needed) and a timestamped subdirectory."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def flatten_config(config: Dict) -> Dict:
    """Recursively flatten nested config dicts for easier logging."""
    flat: Dict[str, object] = {}

    def _recurse(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, val in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                _recurse(new_prefix, val)
        else:
            flat[prefix] = value

    _recurse("", config)
    return flat
