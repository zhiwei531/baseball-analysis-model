"""Path conventions for GVHMR 3D artifacts."""

from __future__ import annotations

from pathlib import Path


def frame_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / condition_id


def frame_manifest_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / f"{condition_id}.csv"


def pose_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses" / clip_id / f"{condition_id}.csv"


def pose3d_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses3d" / clip_id / f"{condition_id}.csv"


def feature3d_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "features3d" / clip_id / f"{condition_id}.csv"


def overlay3d_frame_dir(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays3d" / "frames" / clip_id / condition_id


def overlay3d_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays3d" / f"{clip_id}__{condition_id}.mp4"
