"""High-level orchestration service for sensor read, shm write, and UDS control."""

import logging
import time
from datetime import datetime
from pathlib import Path

from ..config.settings import Settings
from ..io.local_store import LocalStore
from ..io.sensor_client import SensorClient
from ..io.shm_writer import ShmWriter
from ..io.uds_channel import UdsChannel
from ..protocol.messages import ErrorCode, MsgType
from .tactile_qc import compute_tactile_qc, write_tactile_preview_npz
from .state import ServiceState, can_transition


class AcquisitionService:
    """Coordinate sensor IO, state transitions, local storage, shm, and UDS messaging."""

    def __init__(self, settings: Settings):
        """Create service dependencies and initialize runtime counters.

        Args:
            settings: Runtime configuration values shared by all subcomponents.
        """
        self.settings = settings
        if self.settings.target_fps <= 0:
            raise ValueError("settings.target_fps must be > 0")

        self.frame_interval = 1.0 / self.settings.target_fps
        self.state = ServiceState.BOOT

        self.sensor = SensorClient(
            sensor_id_0=settings.sensor_id_0,
            sensor_id_1=settings.sensor_id_1,
            use_gpu=settings.use_gpu,
            save_dir=settings.save_dir,
        )
        self.uds = UdsChannel(
            socket_path=settings.uds_path,
            version=settings.protocol_version,
            recv_timeout_s=settings.uds_recv_timeout_s,
        )
        self.local_store = LocalStore(
            save_dir=Path(settings.save_dir),
            sensor_id_0=settings.sensor_id_0,
            sensor_id_1=settings.sensor_id_1,
        )

        self.shm_writer: ShmWriter | None = None
        self.frame_id = 0
        self.current_demo_tag: str | None = None
        self._next_frame_deadline = time.perf_counter()
        self._running = True

    def run_forever(self) -> None:
        """Run the service main loop until STOP is requested or a fatal error occurs."""
        try:
            self.initialize()
            if self.state == ServiceState.STOPPED:
                return
            self._set_state(ServiceState.WAIT_START)
            self.uds.send_message(MsgType.INIT_READY)
            self.local_store.mark_event("init_ready_ns", time.time_ns())

            while self._running and self.state != ServiceState.STOPPED:
                self._process_control_messages()

                if self.state == ServiceState.COLLECTING:
                    self._collect_once()
                else:
                    time.sleep(0.01)
        finally:
            self.shutdown()

    def initialize(self) -> None:
        """Initialize transport and sensors, then build shm schema from warmup frame."""
        self._set_state(ServiceState.INIT)

        self.uds.start_server()
        self.uds.wait_client()

        try:
            seed_frame = self.sensor.initialize()
            self.shm_writer = ShmWriter.from_frame(self.settings.shm_name, seed_frame)
        except Exception as exc:
            logging.exception("sensor initialization failed")
            self._send_error(ErrorCode.SENSOR_INIT_FAIL, f"sensor init failed: {exc}")
            self._set_state(ServiceState.STOPPED)
            self._running = False

    def _collect_once(self) -> None:
        """Capture one frame and publish it through local store, shm, and UDS."""
        wait_s = self._next_frame_deadline - time.perf_counter()
        if wait_s > 0:
            sleep_s = max(wait_s - 0.001, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

            while time.perf_counter() < self._next_frame_deadline:
                pass

        start_t = time.perf_counter()
        try:
            frame = self.sensor.read_frame(self.frame_id)
            self.local_store.append_frame(frame)
            assert self.shm_writer is not None
            self.shm_writer.write_frame(frame)
            self.uds.send_message(
                MsgType.FRAME_READY,
                frame_id=self.frame_id,
                payload={
                    "timestamp_ns_0": frame.timestamp_ns_0,
                    "timestamp_ns_1": frame.timestamp_ns_1,
                },
            )
            self.frame_id += 1
        except Exception as exc:
            self._send_error(ErrorCode.SENSOR_READ_FAIL, str(exc))
            if self.state == ServiceState.COLLECTING:
                self._set_state(ServiceState.PAUSED)
            return

        self._next_frame_deadline += self.frame_interval
        if self._next_frame_deadline < start_t:
            self._next_frame_deadline = start_t + self.frame_interval

    def _process_control_messages(self) -> None:
        """Poll and handle one control command from the UDS channel if available."""
        msg = self.uds.try_recv_message(max_wait_s=0.0)
        if msg is None:
            return

        msg_type, _frame_id, payload = msg

        if msg_type == MsgType.INIT_REQ:
            self.uds.send_message(MsgType.INIT_READY)
            return

        if msg_type == MsgType.START_REQ:
            if self.state == ServiceState.WAIT_START:
                self._next_frame_deadline = time.perf_counter()
                self.frame_id = 0
                self._set_state(ServiceState.COLLECTING)
                self.current_demo_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.local_store.mark_event("demo_tag", self.current_demo_tag)
                self.local_store.mark_event("start_ns", time.time_ns())
                self.uds.send_message(MsgType.ACK, payload={"cmd": "START_REQ"})
            elif self.state == ServiceState.PAUSED:
                self._next_frame_deadline = time.perf_counter()
                self._set_state(ServiceState.COLLECTING)
                self.local_store.mark_event("resume_ns", time.time_ns())
                self.uds.send_message(MsgType.ACK, payload={"cmd": "START_REQ"})
            else:
                self._send_error(ErrorCode.INVALID_STATE, f"cannot START from {self.state.name}")
            
            return

        if msg_type == MsgType.PAUSE_REQ:
            if self.state == ServiceState.COLLECTING:
                self.local_store.mark_event("pause_ns", time.time_ns())
                self._set_state(ServiceState.PAUSED)
                self.uds.send_message(MsgType.ACK, payload={"cmd": "PAUSE_REQ"})
            else:
                self._send_error(ErrorCode.INVALID_STATE, f"cannot PAUSE from {self.state.name}")
            
            return

        if msg_type == MsgType.DEMO_DONE_REQ:
            if self.state in {ServiceState.COLLECTING, ServiceState.PAUSED}:
                self.local_store.mark_event("demo_done_ns", time.time_ns())
                try:
                    (
                        saved_file,
                        tactile_postcheck,
                        tactile_preview,
                    ) = self._flush_current_demo_with_tactile_metadata()
                    self._set_state(ServiceState.WAIT_START)
                    self._reset_shm_if_available()
                    self.uds.send_message(
                        MsgType.ACK,
                        payload={
                            "cmd": "DEMO_DONE_REQ",
                            "saved_file": saved_file,
                            "xense_tactile_postcheck": tactile_postcheck,
                            "xense_tactile_preview": tactile_preview,
                        },
                    )
                except Exception as exc:
                    self._set_state(ServiceState.PAUSED)
                    self._send_error(ErrorCode.UNKNOWN, f"flush demo failed: {exc}")
            else:
                self._send_error(ErrorCode.INVALID_STATE, f"cannot DEMO_DONE from {self.state.name}")
            
            return

        if msg_type == MsgType.DEMO_DISCARD_REQ:
            if self.state in {ServiceState.COLLECTING, ServiceState.PAUSED}:
                self.local_store.mark_event("demo_discard_ns", time.time_ns())
                self._discard_current_demo()
                self._set_state(ServiceState.WAIT_START)
                try:
                    self._reset_shm_if_available()
                    self.uds.send_message(MsgType.ACK, payload={"cmd": "DEMO_DISCARD_REQ"})
                except Exception as exc:
                    self._set_state(ServiceState.PAUSED)
                    self._send_error(ErrorCode.UNKNOWN, f"reset shm failed: {exc}")
            else:
                self._send_error(ErrorCode.INVALID_STATE, f"cannot DEMO_DISCARD from {self.state.name}")
            
            return

        if msg_type == MsgType.STOP_REQ:
            self.local_store.mark_event("stop_ns", time.time_ns())
            saved_file = None
            try:
                saved_file = self._flush_current_demo()
                self._reset_shm_if_available()
            except Exception as exc:
                self._send_error(ErrorCode.UNKNOWN, f"flush on stop failed: {exc}")
            payload = {"cmd": "STOP_REQ"}
            if saved_file is not None:
                payload["saved_file"] = saved_file
            self.uds.send_message(MsgType.ACK, payload=payload)
            self._set_state(ServiceState.STOPPED)
            self._running = False
            
            return

        self._send_error(ErrorCode.UNKNOWN, f"unsupported msg_type={int(msg_type)} payload={payload}")

    def _set_state(self, next_state: ServiceState) -> None:
        """Switch service state after validating the transition rule."""
        if next_state == self.state:
            return
        if not can_transition(self.state, next_state):
            raise RuntimeError(f"invalid transition: {self.state.name} -> {next_state.name}")
        logging.info("state transition: %s -> %s", self.state.name, next_state.name)
        self.state = next_state

    def _send_error(self, code: ErrorCode, reason: str) -> None:
        """Best-effort ERROR message emission that never raises to caller."""
        try:
            self.uds.send_message(
                MsgType.ERROR,
                frame_id=self.frame_id,
                payload={"code": int(code), "reason": reason},
            )
        except Exception:
            pass

    def _flush_current_demo(self) -> str | None:
        """Persist current buffered demo data and reset the in-memory buffer.

        Returns:
            Saved file name when data exists; otherwise None.
        """
        saved_file, _postcheck, _preview = self._flush_current_demo_with_tactile_metadata()
        return saved_file

    def _flush_current_demo_with_tactile_metadata(
        self,
    ) -> tuple[str | None, dict | None, dict]:
        """Persist buffered demo data and return tactile post-check metadata."""
        if not self.local_store.has_data():
            return None, None, {"ok": False, "path": None, "error": "no tactile data"}
        demo_tag = self.current_demo_tag
        filename = f"data_TAC_{demo_tag}.npy"
        qc_result = compute_tactile_qc(
            self.local_store.data_dict,
            sensor_ids=(self.settings.sensor_id_0, self.settings.sensor_id_1),
            zero_force_mean_tolerance=self.settings.xense_tactile_zero_force_mean_tolerance,
            edge_warning_threshold=self.settings.xense_tactile_edge_warning_threshold,
            edge_window_samples=self.settings.xense_tactile_edge_window_samples,
        )
        self.local_store.flush(filename=filename)
        preview_path = (
            self.settings.tactile_preview_dir
            / f"{Path(filename).stem}_{time.time_ns()}_force_resultant.npz"
        )
        preview_manifest = {"ok": False, "path": str(preview_path), "error": None}
        if qc_result.preview is None:
            preview_manifest["error"] = qc_result.manifest.get("error") or "tactile preview unavailable"
        else:
            try:
                write_tactile_preview_npz(qc_result.preview, preview_path)
                preview_manifest["ok"] = True
            except Exception as exc:
                preview_manifest["error"] = str(exc)
        self.local_store.clear()
        self.current_demo_tag = None
        return filename, qc_result.manifest, preview_manifest

    def _discard_current_demo(self) -> None:
        """Discard current buffered demo data without persisting to disk."""
        self.local_store.clear()
        self.current_demo_tag = None

    def _reset_shm_if_available(self) -> None:
        """Reset shm slots to avoid exposing stale frames to subsequent sessions."""
        if self.shm_writer is not None:
            self.shm_writer.reset_slots()

    def shutdown(self) -> None:
        """Release all resources and flush buffered outputs during service exit."""
        try:
            self.sensor.release()
        except Exception:
            pass

        try:
            if self.shm_writer is not None:
                self.shm_writer.close(unlink=True)
        except Exception:
            pass

        try:
            if self.local_store.has_data():
                self.local_store.flush()
        except Exception:
            pass

        try:
            self.uds.close()
        except Exception:
            pass
