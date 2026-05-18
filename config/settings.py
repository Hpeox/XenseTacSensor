"""Runtime configuration definitions for the acquisition service."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Typed settings container used to configure sensors, UDS, shm, and pacing."""

    sensor_id_0: str = "OG000544"
    sensor_id_1: str = "OG001009"
    use_gpu: bool = True

    uds_path: str = "/tmp/xense_sensor.sock"
    uds_recv_timeout_s: float = 0.2

    shm_name: str = "xense_sensor_frame"
    save_dir: Path = Path(__file__).resolve().parent.parent.parent / "runtime_frames"

    # 连续采集节奏控制；若采集耗时超过该间隔则不会额外 sleep
    target_fps: float = 30.0

    # 二进制协议版本号
    protocol_version: int = 1
