"""Public dataset API with preview / real-contact separation guards."""

from __future__ import annotations

from pathlib import Path

import h5py

from multiarm_sim._multiarm_dataset_base import *  # noqa: F401,F403
from multiarm_sim._multiarm_dataset_base import (
    _attribute_value,
    _dataset,
    _next_trajectory_name,
    _trajectory_names,
    append_multiarm_episode as _append_multiarm_episode,
)


def append_multiarm_episode(
    path,
    episode,
    *,
    final_sim_state,
    success: bool,
):
    """Reject real-contact writes into a file declared preview-only."""

    path = Path(path)
    if path.exists():
        with h5py.File(path, "r") as handle:
            if bool(handle.attrs.get("preview_only", False)):
                raise ValueError(
                    f"{path} is preview-only; refusing real-contact append"
                )
    return _append_multiarm_episode(
        path,
        episode,
        final_sim_state=final_sim_state,
        success=success,
    )
