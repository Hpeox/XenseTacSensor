"""Minimal SHM integration reader for v2 double-buffer protocol."""

import argparse
import contextlib
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Dict

from multiprocessing import resource_tracker

from .io.shm_writer import (
    GLOBAL_HEADER_FMT,
    GLOBAL_HEADER_SIZE,
    SLOT_COUNT,
    SLOT_HEADER_FMT,
    SLOT_HEADER_SIZE,
)


def _read_latest_index(shm: SharedMemory) -> int:
    """Read latest slot index from global header."""
    (latest_index,) = struct.unpack_from(GLOBAL_HEADER_FMT, shm.buf, 0)
    return int(latest_index)


def _slot_base(slot_index: int, slot_stride: int) -> int:
    """Return absolute slot base offset."""
    return GLOBAL_HEADER_SIZE + slot_index * slot_stride


def _read_slot_header(shm: SharedMemory, slot_index: int, slot_stride: int) -> tuple[int, int, int, int]:
    """Read seq/frame_id/timestamps from one slot header."""
    base = _slot_base(slot_index, slot_stride)
    seq, frame_id, timestamp_ns_0, timestamp_ns_1 = struct.unpack_from(SLOT_HEADER_FMT, shm.buf, base)
    return int(seq), int(frame_id), int(timestamp_ns_0), int(timestamp_ns_1)


def _read_consistent_latest_header(
    shm: SharedMemory,
    slot_stride: int,
    max_retries: int,
) -> tuple[int, int, int, int, int]:
    """Read one consistent latest slot header with latest+seq double-check."""
    retries = 0
    while retries < max_retries:
        latest_a = _read_latest_index(shm) % SLOT_COUNT
        seq_a, frame_id, ts0, ts1 = _read_slot_header(shm, latest_a, slot_stride)
        if seq_a % 2 == 1:
            retries += 1
            continue

        seq_b, frame_id_b, ts0_b, ts1_b = _read_slot_header(shm, latest_a, slot_stride)
        latest_b = _read_latest_index(shm) % SLOT_COUNT
        if latest_a == latest_b and seq_a == seq_b and seq_b % 2 == 0:
            if frame_id != frame_id_b or ts0 != ts0_b or ts1 != ts1_b:
                retries += 1
                continue
            return latest_b, frame_id_b, ts0_b, ts1_b, retries

        retries += 1

    raise RuntimeError(f"read failed after retries={max_retries}")


@dataclass
class ShmReaderConfig:
    shm_name: str = "xense_sensor_frame"
    max_retries: int = 200
    target_hz: float = 30.0
    capture_poll_ms: float = 1.0
    capture_timeout_ms: float = 200.0
    dephase_every_n: int = 30
    dephase_ms: float = 1.0
    verbose: bool = True


class ShmReaderRunner:
    """Run SHM reading in background with strict first-frame capture checks."""

    def __init__(self, cfg: ShmReaderConfig):
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._summary: Dict[str, float | int | bool | None] = {}
        self._reset_summary()

    def _reset_summary(self) -> None:
        self._summary = {
            "started_ns": None,
            "stopped_ns": None,
            "success": 0,
            "read_fail": 0,
            "retry_total": 0,
            "slot_switch": 0,
            "frame_regressions": 0,
            "first_frame_id_seen": None,
            "seen_frame_id_0": False,
            "t_seen_frame_id_0_ns": None,
            "capture_timeout": False,
            "capture_last_frame_id": None,
            "period_count": 0,
            "period_mean_ms": 0.0,
            "period_p95_ms": 0.0,
            "period_max_ms": 0.0,
            "avg_retries": 0.0,
            "last_slot": None,
            "last_frame_id": None,
        }

    def start(self) -> None:
        """Start a new reading run."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("reader is already running")
        self._stop_event.clear()
        self._started_event.clear()
        with self._lock:
            self._reset_summary()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_started(self, timeout_s: float = 1.0) -> int | None:
        """Wait until reader thread enters run loop and return started timestamp."""
        if not self._started_event.wait(timeout=timeout_s):
            return None
        with self._lock:
            started_ns = self._summary["started_ns"]
        return int(started_ns) if isinstance(started_ns, int) else None

    def stop(self, timeout_s: float = 2.0) -> Dict[str, float | int | bool | None]:
        """Stop current run and return summary metrics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        return self.summary()

    def summary(self) -> Dict[str, float | int | bool | None]:
        """Return a snapshot of current run metrics."""
        with self._lock:
            return dict(self._summary)

    def _update_summary(self, **kwargs: float | int | bool | None) -> None:
        with self._lock:
            self._summary.update(kwargs)

    def _record_success(
        self,
        slot: int,
        frame_id: int,
        retries: int,
        now_ns: int,
    ) -> None:
        with self._lock:
            self._summary["success"] = int(self._summary["success"]) + 1
            self._summary["retry_total"] = int(self._summary["retry_total"]) + retries

            last_slot = self._summary["last_slot"]
            if isinstance(last_slot, int) and last_slot != slot:
                self._summary["slot_switch"] = int(self._summary["slot_switch"]) + 1

            last_frame = self._summary["last_frame_id"]
            if isinstance(last_frame, int) and frame_id < last_frame:
                self._summary["frame_regressions"] = int(self._summary["frame_regressions"]) + 1

            if self._summary["first_frame_id_seen"] is None:
                self._summary["first_frame_id_seen"] = frame_id
            if frame_id == 0 and not bool(self._summary["seen_frame_id_0"]):
                self._summary["seen_frame_id_0"] = True
                self._summary["t_seen_frame_id_0_ns"] = now_ns

            self._summary["last_slot"] = slot
            self._summary["last_frame_id"] = frame_id

    def _record_period_stats(self, period_ms: list[float]) -> None:
        if not period_ms:
            self._update_summary(period_count=0, period_mean_ms=0.0, period_p95_ms=0.0, period_max_ms=0.0)
            return

        period_ms_sorted = sorted(period_ms)
        p95_idx = min(len(period_ms_sorted) - 1, max(0, int(len(period_ms_sorted) * 0.95) - 1))
        self._update_summary(
            period_count=len(period_ms),
            period_mean_ms=sum(period_ms) / len(period_ms),
            period_p95_ms=period_ms_sorted[p95_idx],
            period_max_ms=max(period_ms),
        )

    def _run(self) -> None:
        started_ns = time.monotonic_ns()
        self._update_summary(started_ns=started_ns)
        self._started_event.set()

        shm = self._attach_reader_shm(self.cfg.shm_name)
        try:
            slot_region_bytes = shm.size - GLOBAL_HEADER_SIZE
            if slot_region_bytes <= 0 or slot_region_bytes % SLOT_COUNT != 0:
                raise RuntimeError(
                    f"invalid shm size={shm.size}, cannot split into {SLOT_COUNT} slot regions"
                )
            slot_stride = slot_region_bytes // SLOT_COUNT

            if self.cfg.verbose:
                print(f"[reader] connected shm={self.cfg.shm_name} slot_stride={slot_stride}")

            capture_deadline_ns = started_ns + int(self.cfg.capture_timeout_ms * 1_000_000)
            capture_sleep_s = max(self.cfg.capture_poll_ms, 0.0) / 1000.0
            period_ns = int(1_000_000_000 / self.cfg.target_hz) if self.cfg.target_hz > 0 else 33_333_333
            dephase_ns = int(max(self.cfg.dephase_ms, 0.0) * 1_000_000)

            steady_phase_started = False
            read_timestamps_ns: list[int] = []
            steady_tick_ns = time.monotonic_ns()
            steady_index = 0

            while not self._stop_event.is_set():
                now_ns = time.monotonic_ns()
                try:
                    slot, frame_id, ts0, ts1, retries = _read_consistent_latest_header(
                        shm=shm,
                        slot_stride=slot_stride,
                        max_retries=self.cfg.max_retries,
                    )
                except RuntimeError:
                    self._update_summary(read_fail=int(self.summary()["read_fail"]) + 1)
                    if not steady_phase_started and now_ns >= capture_deadline_ns:
                        self._update_summary(capture_timeout=True)
                        steady_phase_started = True
                        steady_tick_ns = now_ns
                    if not steady_phase_started:
                        if capture_sleep_s > 0:
                            time.sleep(capture_sleep_s)
                        continue

                    steady_tick_ns += period_ns
                    if now_ns > steady_tick_ns:
                        steady_tick_ns = now_ns
                    sleep_ns = steady_tick_ns - time.monotonic_ns()
                    if sleep_ns > 0:
                        time.sleep(sleep_ns / 1_000_000_000)
                    continue

                self._record_success(slot=slot, frame_id=frame_id, retries=retries, now_ns=now_ns)
                self._update_summary(capture_last_frame_id=frame_id)

                if self.cfg.verbose:
                    print(
                        "[reader]"
                        f" slot={slot} frame_id={frame_id}"
                        f" ts0={ts0} ts1={ts1}"
                        f" retries={retries}"
                    )

                if not steady_phase_started:
                    if frame_id == 0:
                        steady_phase_started = True
                        steady_tick_ns = now_ns
                    elif now_ns >= capture_deadline_ns:
                        self._update_summary(capture_timeout=True)
                        steady_phase_started = True
                        steady_tick_ns = now_ns
                    else:
                        if capture_sleep_s > 0:
                            time.sleep(capture_sleep_s)
                        continue

                read_timestamps_ns.append(now_ns)
                steady_index += 1

                steady_tick_ns += period_ns
                if self.cfg.dephase_every_n > 0 and dephase_ns > 0 and steady_index % self.cfg.dephase_every_n == 0:
                    steady_tick_ns += dephase_ns

                now_after = time.monotonic_ns()
                if steady_tick_ns <= now_after:
                    missed = (now_after - steady_tick_ns) // period_ns + 1
                    steady_tick_ns += missed * period_ns
                sleep_ns = steady_tick_ns - time.monotonic_ns()
                if sleep_ns > 0:
                    time.sleep(sleep_ns / 1_000_000_000)

            period_ms = [
                (read_timestamps_ns[i] - read_timestamps_ns[i - 1]) / 1_000_000.0
                for i in range(1, len(read_timestamps_ns))
            ]
            self._record_period_stats(period_ms)
            summary = self.summary()
            success = int(summary["success"])
            retry_total = int(summary["retry_total"])
            avg_retries = (retry_total / success) if success > 0 else 0.0
            self._update_summary(avg_retries=avg_retries)
        finally:
            shm.close()
            self._update_summary(stopped_ns=time.monotonic_ns())

    @staticmethod
    def _attach_reader_shm(shm_name: str) -> SharedMemory:
        """Attach reader-side shm handle without taking ownership of unlink lifecycle."""
        shm = SharedMemory(name=shm_name, create=False)
        with contextlib.suppress(Exception):
            # Reader is not the owner of shm lifecycle. Unregister avoids
            # resource_tracker warnings when owner process already unlinked it.
            resource_tracker.unregister(shm._name, "shared_memory")
        return shm


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SHM v2 integration reader")
    parser.add_argument("--shm-name", default="xense_sensor_frame", help="Shared memory name")
    parser.add_argument("--duration", type=float, default=5.0, help="Read duration in seconds")
    parser.add_argument("--max-retries", type=int, default=200, help="Max retries per read")
    parser.add_argument("--target-hz", type=float, default=30.0, help="Steady read frequency")
    parser.add_argument("--capture-poll-ms", type=float, default=1.0, help="First-frame capture polling period")
    parser.add_argument("--capture-timeout-ms", type=float, default=200.0, help="Timeout for first-frame capture")
    parser.add_argument("--dephase-every-n", type=int, default=30, help="Apply dephase every N steady reads")
    parser.add_argument("--dephase-ms", type=float, default=1.0, help="Dephase amount in milliseconds")
    parser.add_argument("--strict-frame0", action="store_true", help="Fail if frame_id=0 is not observed")
    args = parser.parse_args()

    runner = ShmReaderRunner(
        ShmReaderConfig(
            shm_name=args.shm_name,
            max_retries=args.max_retries,
            target_hz=args.target_hz,
            capture_poll_ms=args.capture_poll_ms,
            capture_timeout_ms=args.capture_timeout_ms,
            dephase_every_n=args.dephase_every_n,
            dephase_ms=args.dephase_ms,
            verbose=True,
        )
    )

    runner.start()
    time.sleep(max(args.duration, 0.0))
    summary = runner.stop()
    print("[summary]", summary)

    if args.strict_frame0 and not bool(summary["seen_frame_id_0"]):
        raise SystemExit("strict check failed: frame_id=0 was not observed")


if __name__ == "__main__":
    main()
