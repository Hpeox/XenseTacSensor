"""In-memory frame/event accumulation and final numpy persistence."""

import gc
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .sensor_client import FrameData


class LocalStore:
    """Collect per-frame tensors and lifecycle events for offline analysis."""

    def __init__(self, save_dir: Path, sensor_id_0: str, sensor_id_1: str):
        """Prepare in-memory dictionaries and ensure output directory exists."""
        self.save_dir = save_dir
        self.sensor_id_0 = sensor_id_0
        self.sensor_id_1 = sensor_id_1
        self.data_dict: Dict[str, Any] = {"events": {}, "frames_data": {}}

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def append_frame(self, frame: FrameData) -> None:
        """Append one frame under its frame_id key using nested per-frame fields."""
        i = frame.frame_id
        sid0 = self.sensor_id_0
        sid1 = self.sensor_id_1
        frame_key = f"{i:05d}"

        self.data_dict["frames_data"][frame_key] = {
            f"{sid0}_rec": frame.rec_0,
            f"{sid1}_rec": frame.rec_1,
            f"{sid0}_force": frame.force_0,
            f"{sid1}_force": frame.force_1,
            f"{sid0}_force_norm": frame.force_norm_0,
            f"{sid1}_force_norm": frame.force_norm_1,
            f"{sid0}_force_resultant": frame.force_resultant_0,
            f"{sid1}_force_resultant": frame.force_resultant_1,
            f"{sid0}_timestamp_ns": frame.timestamp_ns_0,
            f"{sid1}_timestamp_ns": frame.timestamp_ns_1,
        }

    def mark_event(self, name: str, value: Any) -> None:
        """Store a named lifecycle event marker into the same dictionary."""
        self.data_dict["events"][name] = value

    def flush(self, filename: str = "data_dict.npy") -> None:
        """Persist buffered dictionary to a .npy file with pickle enabled."""
        np.save(self.save_dir / filename, arr=self.data_dict, allow_pickle=True)

    def clear(self) -> None:
        """Clear all buffered frame/event data after a successful flush."""
        self.data_dict.clear()
        gc.collect()
        self.data_dict = {"events": {}, "frames_data": {}}

    def has_data(self) -> bool:
        """Return whether there is buffered frame data pending persistence."""
        return bool(self.data_dict["frames_data"])
