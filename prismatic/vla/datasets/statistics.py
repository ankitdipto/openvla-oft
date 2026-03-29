"""Utilities for serializing dataset statistics for OpenVLA checkpoints."""

import json
from copy import deepcopy

import numpy as np

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Save a checkpoint-local ``dataset_statistics.json`` file."""
    out_path = run_dir / "dataset_statistics.json"
    serializable_stats = deepcopy(dataset_statistics)
    for _, stats in serializable_stats.items():
        for key in ("action", "proprio"):
            if key not in stats:
                continue
            for stat_name, stat_value in stats[key].items():
                if isinstance(stat_value, np.ndarray):
                    stats[key][stat_name] = stat_value.tolist()
        for key in ("num_trajectories", "num_transitions"):
            if key in stats and isinstance(stats[key], np.ndarray):
                stats[key] = stats[key].item()
    with open(out_path, "w") as f_json:
        json.dump(serializable_stats, f_json, indent=2)
    overwatch.info(f"Saved dataset statistics file at path {out_path}")
