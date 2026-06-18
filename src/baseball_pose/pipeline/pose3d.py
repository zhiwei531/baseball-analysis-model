"""Planning and execution for external GVHMR-style 3D lifting stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import nan

from baseball_pose.config import RuntimeConfig, resolve_pose3d_config
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import feature3d_path, frame_manifest_path, pose3d_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.io.pose3d_csv import write_pose3d_records
from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.quality import threshold_for_joint
from baseball_pose.pose3d.external_video_hmr import read_external_video_hmr_records
from baseball_pose.pose.schema import pose_score


@dataclass(frozen=True)
class Pose3DPlan:
    clip_id: str
    input_condition_id: str
    source_condition_id: str
    output_condition_id: str
    input_pose_path: Path
    input_frames_path: Path
    output_pose3d_path: Path
    output_feature3d_path: Path
    backend: str
    root_joint: str


def build_pose3d_plan(
    clip: ClipMetadata,
    config: RuntimeConfig,
    *,
    input_condition_id: str,
    output_condition_id: str | None = None,
) -> Pose3DPlan:
    pose3d_config = resolve_pose3d_config(config.raw, clip.clip_id)
    source_condition_id = _source_condition_for_3d(input_condition_id)
    derived_output_condition = output_condition_id or f"{input_condition_id}_3d"
    return Pose3DPlan(
        clip_id=clip.clip_id,
        input_condition_id=input_condition_id,
        source_condition_id=source_condition_id,
        output_condition_id=derived_output_condition,
        input_pose_path=pose_path(config.data_dir, clip.clip_id, input_condition_id),
        input_frames_path=frame_manifest_path(config.data_dir, clip.clip_id, source_condition_id),
        output_pose3d_path=pose3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        output_feature3d_path=feature3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        backend=str(pose3d_config.get("backend", "gvhmr")),
        root_joint=str(pose3d_config.get("root_joint", "pelvis_center")),
    )


def lift_pose_sequence(plan: Pose3DPlan, clip: ClipMetadata, config: RuntimeConfig) -> int:
    """Import external GVHMR/HMR 3D results into the project CSV contract."""

    pose3d_config = resolve_pose3d_config(config.raw, clip.clip_id)
    if plan.backend not in {"external_video_hmr", "gvhmr", "wham"}:
        raise NotImplementedError(f"Unsupported configured 3D backend: {plan.backend}")

    records = _lift_external_video_hmr(plan, clip, config, pose3d_config)
    if plan.input_pose_path.exists():
        records = _gate_pose3d_with_2d_prior(
            records,
            pose2d_path=plan.input_pose_path,
            pose3d_config=pose3d_config,
            hard_gate=bool(pose3d_config.get("gate_with_2d_prior", False)),
        )
    write_pose3d_records(plan.output_pose3d_path, records)
    return len(records)


def _lift_external_video_hmr(
    plan: Pose3DPlan,
    clip: ClipMetadata,
    config: RuntimeConfig,
    pose3d_config: dict[str, object],
):
    frames = _read_external_3d_timeline(plan)
    result_path = _external_video_hmr_result_path(
        pose3d_config,
        config=config,
        clip_id=clip.clip_id,
        input_condition_id=plan.input_condition_id,
        source_condition_id=plan.source_condition_id,
        output_condition_id=plan.output_condition_id,
        backend=plan.backend,
    )
    return read_external_video_hmr_records(
        result_path,
        clip_id=clip.clip_id,
        condition_id=plan.output_condition_id,
        frames=frames,
        backend_name=plan.backend,
        scale_mode=str(pose3d_config.get("external_scale_mode", "external_video_hmr")),
        joint_names=_string_list(pose3d_config.get("external_joint_names")),
        joint_name_map=_string_map(pose3d_config.get("external_joint_name_map")),
    )


def _read_external_3d_timeline(plan: Pose3DPlan) -> list[FrameRecord]:
    if plan.input_frames_path.exists():
        return read_frame_records(plan.input_frames_path)
    if not plan.input_pose_path.exists():
        raise FileNotFoundError(
            f"External 3D import requires either sampled frames or input 2D poses: "
            f"{plan.input_frames_path} / {plan.input_pose_path}"
        )
    records_by_frame = {}
    for record in read_pose_records(plan.input_pose_path):
        records_by_frame.setdefault(record.frame_index, record.timestamp_sec)
    return [
        FrameRecord(
            clip_id=plan.clip_id,
            condition_id=plan.source_condition_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_path=Path(""),
            width=None,
            height=None,
        )
        for frame_index, timestamp_sec in sorted(records_by_frame.items())
    ]


def _external_video_hmr_result_path(
    pose3d_config: dict[str, object],
    *,
    config: RuntimeConfig,
    clip_id: str,
    input_condition_id: str,
    source_condition_id: str,
    output_condition_id: str,
    backend: str,
) -> Path:
    template = str(
        pose3d_config.get(
            "external_result_path",
            "{data_dir}/external_pose3d/{backend}/{clip_id}.csv",
        )
    )
    return Path(
        template.format(
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            clip_id=clip_id,
            input_condition_id=input_condition_id,
            source_condition_id=source_condition_id,
            output_condition_id=output_condition_id,
            backend=backend,
        )
    )


def _string_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("pose3d.external_joint_names must be a list.")
    return [str(item) for item in value]


def _string_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pose3d.external_joint_name_map must be a mapping.")
    return {str(key): str(mapped) for key, mapped in value.items()}


def lift_pose_sequence_placeholder(plan: Pose3DPlan) -> None:
    """Backward-compatible alias retained for the initial 3D planning commit."""

    raise NotImplementedError(
        "3D lifting placeholder has been superseded by lift_pose_sequence(). Planned input:"
        f" {plan.input_frames_path} -> {plan.output_pose3d_path}"
    )


def _source_condition_for_3d(input_condition_id: str) -> str:
    if input_condition_id.endswith("_smooth"):
        return input_condition_id[: -len("_smooth")]
    return input_condition_id


def _gate_pose3d_with_2d_prior(
    records,
    *,
    pose2d_path: Path,
    pose3d_config: dict[str, object],
    hard_gate: bool = True,
):
    """Use the cleaned 2D pipeline as a trust prior for 3D joints."""

    threshold_config = pose3d_config.get("confidence_thresholds", {})
    default_threshold = float(pose3d_config.get("confidence_threshold", 0.5))
    pose2d_records = read_pose_records(pose2d_path)
    pose2d_by_key = {
        (record.frame_index, record.joint_name): record
        for record in pose2d_records
    }
    gated = []
    for record in records:
        prior = pose2d_by_key.get((record.frame_index, record.joint_name))
        if prior is None:
            gated.append(record)
            continue
        threshold = threshold_for_joint(record.joint_name, default_threshold, threshold_config if isinstance(threshold_config, dict) else {})
        prior_score = pose_score(prior)
        should_reject = prior.x is None or prior.y is None or (prior_score is not None and prior_score < threshold)
        if hard_gate and should_reject:
            gated.append(
                record.__class__(
                    clip_id=record.clip_id,
                    condition_id=record.condition_id,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    joint_name=record.joint_name,
                    x_3d=nan,
                    y_3d=nan,
                    z_3d=nan,
                    scale_mode=record.scale_mode,
                    lift_backend=record.lift_backend,
                    input_quality_score=prior_score,
                )
            )
            continue
        merged_score = record.input_quality_score
        if prior_score is not None and merged_score is not None:
            merged_score = min(prior_score, merged_score)
        elif prior_score is not None:
            merged_score = prior_score
        gated.append(
            record.__class__(
                clip_id=record.clip_id,
                condition_id=record.condition_id,
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                joint_name=record.joint_name,
                x_3d=record.x_3d,
                y_3d=record.y_3d,
                z_3d=record.z_3d,
                scale_mode=record.scale_mode,
                lift_backend=record.lift_backend,
                input_quality_score=merged_score,
            )
        )
    return gated
