"""Application entrypoint for the tactile acquisition service."""

import argparse
import logging

from .config.settings import Settings
from .core.service import AcquisitionService


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments used to override runtime settings."""
    parser = argparse.ArgumentParser(description="Xense tactile acquisition service")
    parser.add_argument("--uds-path", default=None, help="UDS socket path")
    parser.add_argument("--shm-name", default=None, help="Shared memory name")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS")
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

    service = AcquisitionService(settings)
    service.run_forever()


if __name__ == "__main__":
    main()
