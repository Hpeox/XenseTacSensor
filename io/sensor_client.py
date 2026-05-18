"""Sensor SDK wrapper for two-device initialization and serial frame capture."""

import time
from dataclasses import dataclass
from typing import Any, Dict
from xensesdk import Sensor
from ..sdk_patch.xense_patch import patch_xense_diff_model
from ..config.settings import Settings


@dataclass
class FrameData:
    """Container for a synchronized frame payload from both sensors."""

    frame_id: int
    timestamp_ns_0: int
    timestamp_ns_1: int
    rec_0: Any
    force_0: Any
    force_norm_0: Any
    force_resultant_0: Any
    rec_1: Any
    force_1: Any
    force_norm_1: Any
    force_resultant_1: Any


class SensorClient:
    """Encapsulate dual-sensor lifecycle and per-frame read operations."""

    def __init__(self, sensor_id_0: str, sensor_id_1: str, use_gpu: bool = True):
        """Store sensor identifiers and runtime flags without touching hardware."""
        self.sensor_id_0 = sensor_id_0
        self.sensor_id_1 = sensor_id_1
        self.use_gpu = use_gpu

        self._sensor_0 = None
        self._sensor_1 = None

    def initialize(self) -> FrameData:
        """Create sensors and return one warmup frame.

        The returned warmup frame uses frame_id = -1 and should only be used for
        schema probing (for example shm layout), not for persistence or publish.
        """

        self._sensor_0 = Sensor.create(self.sensor_id_0, use_gpu=self.use_gpu)
        self._sensor_1 = Sensor.create(self.sensor_id_1, use_gpu=self.use_gpu)

        # patch diff model for better performance and stability
        patch_xense_diff_model(self._sensor_0)
        patch_xense_diff_model(self._sensor_1)

        # make a folder named current timestamp (YYMMDD_HHMMSS) in Settings.save_dir to save timestamps
        timestamp_dir = (Settings.save_dir / time.strftime("%Y%m%d_%H%M%S"))
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        self._sensor_0.exportRuntimeConfig(timestamp_dir)
        self._sensor_1.exportRuntimeConfig(timestamp_dir)

        # retuen with warmup
        return self.read_frame(frame_id=-1)

    def read_frame(self, frame_id: int) -> FrameData:
        """Read one frame sequentially from sensor_0 then sensor_1.

        Args:
            frame_id: Monotonic frame sequence number assigned by service.

        Returns:
            FrameData containing per-sensor timestamps and all selected outputs.
        """
        if self._sensor_0 is None or self._sensor_1 is None:
            raise RuntimeError("sensor client not initialized")

        timestamp_ns_0 = time.time_ns()
        rec_0, force_0, force_norm_0, force_resultant_0 = self._sensor_0.selectSensorInfo(
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceNorm,
            Sensor.OutputType.ForceResultant,
        )
        timestamp_ns_1 = time.time_ns()
        rec_1, force_1, force_norm_1, force_resultant_1 = self._sensor_1.selectSensorInfo(
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Force,
            Sensor.OutputType.ForceNorm,
            Sensor.OutputType.ForceResultant,
        )

        return FrameData(
            frame_id=frame_id,
            timestamp_ns_0=timestamp_ns_0,
            timestamp_ns_1=timestamp_ns_1,
            rec_0=rec_0,
            force_0=force_0,
            force_norm_0=force_norm_0,
            force_resultant_0=force_resultant_0,
            rec_1=rec_1,
            force_1=force_1,
            force_norm_1=force_norm_1,
            force_resultant_1=force_resultant_1,
        )

    def release(self) -> None:
        """Release hardware resources for both sensors safely."""
        if self._sensor_0 is not None:
            self._sensor_0.release()
            self._sensor_0 = None
        if self._sensor_1 is not None:
            self._sensor_1.release()
            self._sensor_1 = None

    @staticmethod
    def frame_to_dict(frame: FrameData) -> Dict[str, Any]:
        """Convert FrameData into a plain dictionary representation."""
        return {
            "frame_id": frame.frame_id,
            "timestamp_ns_0": frame.timestamp_ns_0,
            "timestamp_ns_1": frame.timestamp_ns_1,
            "rec_0": frame.rec_0,
            "force_0": frame.force_0,
            "force_norm_0": frame.force_norm_0,
            "force_resultant_0": frame.force_resultant_0,
            "rec_1": frame.rec_1,
            "force_1": frame.force_1,
            "force_norm_1": frame.force_norm_1,
            "force_resultant_1": frame.force_resultant_1,
        }
