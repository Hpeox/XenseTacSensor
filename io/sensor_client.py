"""Sensor SDK wrapper for two-device initialization and parallel frame capture."""

import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Dict

from ..config.settings import Settings


XENSE_ENV_TO_SDK_VERSION = {
    "Xense310": "1.x",
    "xense2_bak": "2.0",
    "xense2": "2.0.1",
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
    if sdk_version in {"2.0", "2.0.1"}:
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


def _select_sensor_payload(sensor: Any, Sensor: Any) -> tuple[int, tuple[Any, Any, Any, Any]]:
    """Return a timestamp and selected sensor outputs."""
    timestamp_ns = time.time_ns()
    payload = sensor.selectSensorInfo(
        Sensor.OutputType.Rectify,
        Sensor.OutputType.Force,
        Sensor.OutputType.ForceNorm,
        Sensor.OutputType.ForceResultant,
    )
    return timestamp_ns, payload


def _create_worker_sensor(sensor_id: str, use_gpu: bool, sdk_version: str, Sensor: Any) -> Any:
    """Create one SDK sensor with version-specific construction and patching."""
    if sdk_version == "1.x":
        sensor = Sensor.create(sensor_id, use_gpu=use_gpu)

        from ..sdk_patch.xense_patch import patch_xense_diff_model

        patch_xense_diff_model(sensor)
        return sensor

    if sdk_version not in {"2.0", "2.0.1"}:
        raise RuntimeError(f"unsupported Xense SDK version: {sdk_version}")

    return Sensor.create(sensor_id)


def _runtime_config_dir(save_dir: Path) -> Path:
    """Return the timestamped directory used for SDK runtime config exports."""
    timestamp_dir = save_dir / time.strftime("%Y%m%d_%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    return timestamp_dir


def _sensor_worker_main(
    sensor_index: int,
    sensor_id: str,
    use_gpu: bool,
    runtime_config_dir: Path,
    command_queue: Any,
    result_queue: Any,
) -> None:
    """Own one SDK sensor and serve frame-read commands from the parent process."""
    sensor = None
    Sensor = None
    try:
        sdk_version, Sensor = load_sensor_api()
        sensor = _create_worker_sensor(sensor_id, use_gpu, sdk_version, Sensor)
        sensor.exportRuntimeConfig(runtime_config_dir)
        timestamp_ns, payload = _select_sensor_payload(sensor, Sensor)
        result_queue.put(
            {
                "type": "ready",
                "sensor_index": sensor_index,
                "timestamp_ns": timestamp_ns,
                "payload": payload,
            }
        )

        while True:
            command = command_queue.get()
            command_type = command[0]
            if command_type == "stop":
                return
            if command_type != "read":
                raise RuntimeError(f"unsupported worker command: {command_type}")

            frame_id = command[1]
            timestamp_ns, payload = _select_sensor_payload(sensor, Sensor)
            result_queue.put(
                {
                    "type": "frame",
                    "sensor_index": sensor_index,
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "payload": payload,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "sensor_index": sensor_index,
                "error": repr(exc),
            }
        )
    finally:
        if sensor is not None:
            sensor.release()


def _frame_from_results(frame_id: int, result_0: dict[str, Any], result_1: dict[str, Any]) -> FrameData:
    """Build the public FrameData object from two worker result messages."""
    rec_0, force_0, force_norm_0, force_resultant_0 = result_0["payload"]
    rec_1, force_1, force_norm_1, force_resultant_1 = result_1["payload"]

    return FrameData(
        frame_id=frame_id,
        timestamp_ns_0=result_0["timestamp_ns"],
        timestamp_ns_1=result_1["timestamp_ns"],
        rec_0=rec_0,
        force_0=force_0,
        force_norm_0=force_norm_0,
        force_resultant_0=force_resultant_0,
        rec_1=rec_1,
        force_1=force_1,
        force_norm_1=force_norm_1,
        force_resultant_1=force_resultant_1,
    )


class SensorClient:
    """Encapsulate dual-sensor lifecycle and per-frame read operations."""

    def __init__(
        self,
        sensor_id_0: str,
        sensor_id_1: str,
        use_gpu: bool = True,
        worker_start_timeout_s: float = 60.0,
        read_timeout_s: float = 5.0,
        worker_stop_timeout_s: float = 5.0,
        save_dir: Path | None = None,
    ):
        """Store sensor identifiers and runtime flags without touching hardware."""
        self.sensor_id_0 = sensor_id_0
        self.sensor_id_1 = sensor_id_1
        self.use_gpu = use_gpu
        self.worker_start_timeout_s = worker_start_timeout_s
        self.read_timeout_s = read_timeout_s
        self.worker_stop_timeout_s = worker_stop_timeout_s
        self.save_dir = Path(Settings.save_dir if save_dir is None else save_dir)

        self._sensor_0 = None
        self._sensor_1 = None
        self._sensor_api = None
        self._ctx = None
        self._command_queues: list[Any] = []
        self._result_queue = None
        self._worker_processes: list[Any] = []
        self._initialized = False

    def initialize(self) -> FrameData:
        """Start sensor workers and return one warmup frame.

        The returned warmup frame uses frame_id = -1 and should only be used for
        schema probing (for example shm layout), not for persistence or publish.
        """
        self.release()
        self._ctx = mp.get_context("spawn")
        self._result_queue = self._ctx.Queue()
        self._command_queues = []
        self._worker_processes = []
        runtime_config_dir = _runtime_config_dir(self.save_dir)

        try:
            for sensor_index, sensor_id in enumerate((self.sensor_id_0, self.sensor_id_1)):
                command_queue = self._ctx.Queue()
                process = self._ctx.Process(
                    target=_sensor_worker_main,
                    args=(
                        sensor_index,
                        sensor_id,
                        self.use_gpu,
                        runtime_config_dir,
                        command_queue,
                        self._result_queue,
                    ),
                    name=f"xense-sensor-{sensor_index}",
                )
                self._command_queues.append(command_queue)
                self._worker_processes.append(process)
                process.start()

            ready_results = self._wait_for_results(
                expected_type="ready",
                expected_frame_id=None,
                timeout_s=self.worker_start_timeout_s,
            )
        except Exception:
            self.release()
            raise

        self._initialized = True
        self._sensor_api = "worker"
        return _frame_from_results(frame_id=-1, result_0=ready_results[0], result_1=ready_results[1])

    def read_frame(self, frame_id: int) -> FrameData:
        """Read one frame from both sensor workers in parallel.

        Args:
            frame_id: Monotonic frame sequence number assigned by service.

        Returns:
            FrameData containing per-sensor timestamps and all selected outputs.
        """
        if not self._initialized or self._result_queue is None or len(self._command_queues) != 2:
            raise RuntimeError("sensor client not initialized")

        for command_queue in self._command_queues:
            command_queue.put(("read", frame_id))

        frame_results = self._wait_for_results(
            expected_type="frame",
            expected_frame_id=frame_id,
            timeout_s=self.read_timeout_s,
        )
        return _frame_from_results(frame_id=frame_id, result_0=frame_results[0], result_1=frame_results[1])

    def _wait_for_results(
        self,
        expected_type: str,
        expected_frame_id: int | None,
        timeout_s: float,
    ) -> dict[int, dict[str, Any]]:
        """Wait until both workers publish one matching message."""
        assert self._result_queue is not None
        deadline = time.monotonic() + timeout_s
        results: dict[int, dict[str, Any]] = {}

        while len(results) < 2:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                self._raise_if_worker_exited()
                raise TimeoutError(f"timed out waiting for {expected_type} from sensor workers")

            try:
                message = self._result_queue.get(timeout=min(remaining_s, 0.1))
            except Empty:
                self._raise_if_worker_exited()
                continue

            message_type = message.get("type")
            sensor_index = message.get("sensor_index")
            if message_type == "error":
                raise RuntimeError(f"sensor worker {sensor_index} failed: {message.get('error')}")
            if message_type != expected_type:
                raise RuntimeError(f"unexpected worker message type: {message_type}")
            if sensor_index not in (0, 1):
                raise RuntimeError(f"unexpected worker sensor_index: {sensor_index}")
            if expected_frame_id is not None and message.get("frame_id") != expected_frame_id:
                raise RuntimeError(
                    f"unexpected worker frame_id from sensor {sensor_index}: "
                    f"{message.get('frame_id')} != {expected_frame_id}"
                )
            results[sensor_index] = message

        return results

    def _raise_if_worker_exited(self) -> None:
        """Raise if any worker has exited before publishing the expected result."""
        for sensor_index, process in enumerate(self._worker_processes):
            if not process.is_alive() and process.exitcode is not None:
                raise RuntimeError(
                    f"sensor worker {sensor_index} exited unexpectedly with code {process.exitcode}"
                )

    def release(self) -> None:
        """Stop worker processes and release parent-side resources safely."""
        for command_queue in self._command_queues:
            try:
                command_queue.put(("stop", None))
            except Exception:
                pass

        for process in self._worker_processes:
            try:
                process.join(timeout=self.worker_stop_timeout_s)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            except Exception:
                pass

        for queue in self._command_queues:
            self._close_queue(queue)
        if self._result_queue is not None:
            self._close_queue(self._result_queue)

        self._sensor_0 = None
        self._sensor_1 = None
        self._sensor_api = None
        self._ctx = None
        self._command_queues = []
        self._result_queue = None
        self._worker_processes = []
        self._initialized = False

    @staticmethod
    def _close_queue(queue: Any) -> None:
        """Best-effort multiprocessing queue cleanup."""
        try:
            queue.close()
        except Exception:
            pass
        try:
            queue.join_thread()
        except Exception:
            pass

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
