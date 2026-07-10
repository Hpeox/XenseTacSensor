"""Application entrypoint for the tactile acquisition service."""

import argparse
import logging
from pathlib import Path

from .config.settings import Settings
from .core.service import AcquisitionService
from .io.sensor_client import MockSensorClient


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments used to override runtime settings."""
    parser = argparse.ArgumentParser(description="Xense tactile acquisition service")
    parser.add_argument("--uds-path", default=None, help="UDS socket path")
    parser.add_argument("--shm-name", default=None, help="Shared memory name")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for saved runtime frames")
    parser.add_argument("--tactile-preview-dir", type=Path, default=None, help="Directory for lightweight tactile preview files")
    parser.add_argument("--xense-tactile-zero-force-mean-tolerance", type=float, default=None)
    parser.add_argument("--xense-tactile-edge-warning-threshold", type=float, default=None)
    parser.add_argument("--xense-tactile-edge-window-samples", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock tactile frames without SDK or hardware")
    return parser.parse_args()


def main() -> None:
    """Configure logging, build settings, and run the acquisition service."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    settings = Settings()
    if args.uds_path:
        settings.uds_path = args.uds_path
    if args.shm_name:
        settings.shm_name = args.shm_name
    if args.fps is not None:
        settings.target_fps = args.fps
    if args.save_dir is not None:
        settings.save_dir = args.save_dir.expanduser().resolve()
    if args.tactile_preview_dir is not None:
        settings.tactile_preview_dir = args.tactile_preview_dir.expanduser().resolve()
    if args.xense_tactile_zero_force_mean_tolerance is not None:
        settings.xense_tactile_zero_force_mean_tolerance = args.xense_tactile_zero_force_mean_tolerance
    if args.xense_tactile_edge_warning_threshold is not None:
        settings.xense_tactile_edge_warning_threshold = args.xense_tactile_edge_warning_threshold
    if args.xense_tactile_edge_window_samples is not None:
        settings.xense_tactile_edge_window_samples = args.xense_tactile_edge_window_samples

    sensor_client = MockSensorClient() if args.mock else None
    service = AcquisitionService(settings, sensor_client=sensor_client)
    service.run_forever()


if __name__ == "__main__":
    main()
