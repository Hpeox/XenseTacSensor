"""Tactile force-resultant quality checks and lightweight preview export."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TactilePreview:
    """Small force-resultant payload used by MainController preview plotting."""

    sensor_ids: tuple[str, str]
    frame_index: np.ndarray
    force_resultant: np.ndarray
    edge_warning: np.ndarray
    edge_max: np.ndarray


@dataclass(frozen=True)
class TactileQcResult:
    """Post-check manifest plus optional preview arrays."""

    manifest: dict[str, Any]
    preview: TactilePreview | None


def _json_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _sorted_frame_keys(frames_data: dict[str, Any]) -> list[str]:
    try:
        return sorted(frames_data, key=lambda key: int(key))
    except ValueError:
        return sorted(frames_data)


def _load_force_resultants(
    frames_data: dict[str, Any],
    sensor_ids: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    frame_keys = _sorted_frame_keys(frames_data)
    if not frame_keys:
        raise ValueError('no tactile frames')

    per_sensor: list[list[np.ndarray]] = [[], []]
    frame_index: list[int] = []
    for frame_key in frame_keys:
        frame = frames_data[frame_key]
        if not isinstance(frame, dict):
            raise ValueError(f'frame {frame_key!r} is not a dict')
        frame_index.append(int(frame_key) if str(frame_key).isdigit() else len(frame_index))
        for sensor_index, sensor_id in enumerate(sensor_ids):
            field = f'{sensor_id}_force_resultant'
            if field not in frame:
                raise ValueError(f'missing tactile field: {field}')
            values = np.asarray(frame[field], dtype=np.float64)
            if values.shape != (6,):
                raise ValueError(f'{field} must have shape (6,), got {values.shape}')
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{field} contains non-finite values')
            per_sensor[sensor_index].append(values)

    force_resultant = np.stack(
        [np.stack(sensor_values, axis=0) for sensor_values in per_sensor],
        axis=0,
    )
    return np.asarray(frame_index, dtype=np.int64), force_resultant


def _edge_max(force_resultant: np.ndarray, window_samples: int) -> float:
    sample_count = force_resultant.shape[0]
    if sample_count == 0:
        raise ValueError('empty force_resultant')
    window = max(int(window_samples), 1)
    head = force_resultant[: min(window, sample_count)]
    tail = force_resultant[max(sample_count - window, 0) :]
    head_channel_mean = np.mean(np.abs(head), axis=0)
    tail_channel_mean = np.mean(np.abs(tail), axis=0)
    return _json_float(np.max(np.concatenate([head_channel_mean, tail_channel_mean])))


def compute_tactile_qc(
    data_dict: dict[str, Any],
    *,
    sensor_ids: tuple[str, str],
    zero_force_mean_tolerance: float,
    edge_warning_threshold: float,
    edge_window_samples: int,
) -> TactileQcResult:
    """Compute tactile post-check summary and preview arrays from LocalStore data."""
    try:
        frames_data = data_dict.get('frames_data')
        if not isinstance(frames_data, dict):
            raise ValueError('frames_data is not a dict')
        frame_index, force_resultant = _load_force_resultants(frames_data, sensor_ids)

        sensors: list[dict[str, Any]] = []
        edge_warning_values: list[bool] = []
        edge_max_values: list[float] = []
        zero_force_values: list[bool] = []
        warnings: list[str] = []
        for sensor_index, sensor_id in enumerate(sensor_ids):
            sensor_force = force_resultant[sensor_index]
            force_norm = np.linalg.norm(sensor_force, axis=1)
            mean_norm = _json_float(np.mean(force_norm))
            zero_force = bool(mean_norm <= float(zero_force_mean_tolerance))
            edge_max = _edge_max(sensor_force, edge_window_samples)
            edge_warning = bool(edge_max > float(edge_warning_threshold))
            if edge_warning:
                warnings.append(
                    f'{sensor_id} tactile edge warning: edge_max={edge_max:.6g} '
                    f'exceeds threshold {float(edge_warning_threshold):.6g}'
                )
            sensors.append(
                {
                    'sensor_id': sensor_id,
                    'samples': int(sensor_force.shape[0]),
                    'mean_norm': mean_norm,
                    'zero_force': zero_force,
                    'edge_max': edge_max,
                    'edge_warning': edge_warning,
                }
            )
            edge_warning_values.append(edge_warning)
            edge_max_values.append(edge_max)
            zero_force_values.append(zero_force)

        one_zero_force = sum(1 for value in zero_force_values if value) == 1
        if one_zero_force:
            warnings.append('exactly one tactile sensor is zero-force')

        manifest = {
            'ok': not one_zero_force,
            'has_warning': any(edge_warning_values),
            'warnings': warnings,
            'zero_force_mean_tolerance': float(zero_force_mean_tolerance),
            'edge_warning_threshold': float(edge_warning_threshold),
            'edge_window_samples': int(edge_window_samples),
            'sensors': sensors,
        }
        preview = TactilePreview(
            sensor_ids=sensor_ids,
            frame_index=frame_index,
            force_resultant=force_resultant,
            edge_warning=np.asarray(edge_warning_values, dtype=np.bool_),
            edge_max=np.asarray(edge_max_values, dtype=np.float64),
        )
        return TactileQcResult(manifest=manifest, preview=preview)
    except Exception as exc:
        return TactileQcResult(
            manifest={
                'ok': False,
                'has_warning': False,
                'warnings': [],
                'zero_force_mean_tolerance': float(zero_force_mean_tolerance),
                'edge_warning_threshold': float(edge_warning_threshold),
                'edge_window_samples': int(edge_window_samples),
                'sensors': [],
                'error': str(exc),
            },
            preview=None,
        )


def write_tactile_preview_npz(preview: TactilePreview, output_path: Path) -> Path:
    """Atomically write a small tactile preview archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f'.{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}'
    )
    try:
        with temporary_path.open('wb') as fp:
            np.savez_compressed(
                fp,
                sensor_ids=np.asarray(preview.sensor_ids, dtype='<U64'),
                frame_index=preview.frame_index,
                force_resultant=preview.force_resultant,
                edge_warning=preview.edge_warning,
                edge_max=preview.edge_max,
            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
