"""Sensor SDK wrapper for two-device initialization and serial frame capture."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..config.settings import Settings


XENSE_ENV_TO_SDK_VERSION = {
    "Xense310": "1.x",
    "xense2": "2.0",
}


def _xense_sdk_from_executable(executable: str | None = None) -> tuple[str, str]:
    """Return the conda env and SDK version encoded in sys.executable."""
    exe = Path(sys.executable if executable is None else executable).resolve()
    parts = exe.parts
    env_name = None
    for index, part in enumerate(parts):
        if part == "envs" and index + 1 < len(parts):
            env_name = parts[index + 1]
            break
    if env_name is None:
        raise RuntimeError(f"cannot determine Xense conda env from sys.executable={exe}")
    try:
        return env_name, XENSE_ENV_TO_SDK_VERSION[env_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(XENSE_ENV_TO_SDK_VERSION))
        raise RuntimeError(
            f"unsupported Xense conda env {env_name!r}; expected one of: {allowed}"
        ) from exc


def load_sensor_api() -> tuple[str, Any]:
    """Load the SDK with version-specific patch ordering."""
    env_name, sdk_version = _xense_sdk_from_executable()
    print(f"[xense_sensor] conda env={env_name} sdk_version={sdk_version}")
    if sdk_version == "2.0":
        from ..sdk_patch import xense2_ort_patch  # noqa: F401
        from xensesdk import Sensor

        return sdk_version, Sensor

    from xensesdk import Sensor

    return sdk_version, Sensor


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

    def __init__(
        self,
        sensor_id_0: str,
        sensor_id_1: str,
        use_gpu: bool = True,
        save_dir: Path | None = None,
    ):
        """Store sensor identifiers and runtime flags without touching hardware."""
        self.sensor_id_0 = sensor_id_0
        self.sensor_id_1 = sensor_id_1
        self.use_gpu = use_gpu
        self.save_dir = Path(Settings.save_dir if save_dir is None else save_dir)

        self._sensor_0 = None
        self._sensor_1 = None
        self._sensor_api = None

    def initialize(self) -> FrameData:
        """Create sensors and return one warmup frame.

        The returned warmup frame uses frame_id = -1 and should only be used for
        schema probing (for example shm layout), not for persistence or publish.
        """

        sdk_version, Sensor = load_sensor_api()
        self._sensor_api = Sensor

        if sdk_version == "1.x":
            self._sensor_0 = Sensor.create(self.sensor_id_0, use_gpu=self.use_gpu)
            self._sensor_1 = Sensor.create(self.sensor_id_1, use_gpu=self.use_gpu)

            from ..sdk_patch.xense_patch import patch_xense_diff_model

            patch_xense_diff_model(self._sensor_0)
            patch_xense_diff_model(self._sensor_1)
        else:
            self._sensor_0 = Sensor.create(self.sensor_id_0)
            self._sensor_1 = Sensor.create(self.sensor_id_1)

        # Make a timestamped directory under the configured runtime frame root.
        timestamp_dir = self.save_dir / time.strftime("%Y%m%d_%H%M%S")
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
        if self._sensor_0 is None or self._sensor_1 is None or self._sensor_api is None:
            raise RuntimeError("sensor client not initialized")

        Sensor = self._sensor_api
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
        self._sensor_api = None

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
