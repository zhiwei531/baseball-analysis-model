"""Pipeline adapters for Jiaming's SRS 2D pose extractor."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import write_frame_records
from baseball_pose.io.paths import frame_dir, frame_manifest_path, pose_path
from baseball_pose.io.pose_csv import write_pose_records
from baseball_pose.io.video import FrameRecord
from baseball_pose.pipeline.overlays3d import render_pose3d_overlays
from baseball_pose.pipeline.pose3d import build_pose3d_plan, lift_pose_sequence
from baseball_pose.pipeline.postprocess3d import smooth_pose3d_files
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.srs2d import extractor


@dataclass(frozen=True)
class SRS2DResult:
    clip_id: str
    condition_id: str
    output_dir: Path
    landmark_csv: Path
    frame_quality_csv: Path
    stable_pose_video: Path
    quality_boxes_video: Path
    comparison_video: Path
    project_pose_csv: Path
    frame_manifest_csv: Path
    frame_count: int


@dataclass(frozen=True)
class TwoDThreeDResult:
    srs2d: SRS2DResult
    pose3d_csv: Path | None
    smoothed_pose3d_csv: Path | None
    overlay3d_video: Path | None


def run_srs_2d_pipeline(
    input_video: str | Path,
    *,
    config: RuntimeConfig,
    clip_id: str | None = None,
    condition_id: str = "srs_2d_pose",
    output_prefix: str | None = None,
    max_frames: int = 0,
) -> SRS2DResult:
    """Run Jiaming's SRS 2D extractor and emit project-compatible CSVs."""

    source = Path(input_video)
    resolved_clip_id = clip_id or _safe_clip_id(source.stem)
    prefix = output_prefix or resolved_clip_id
    srs_output_dir = Path(config.output_dir) / "srs2d" / resolved_clip_id
    srs_output_dir.mkdir(parents=True, exist_ok=True)

    args = _extractor_args(
        input_video=source,
        output_dir=srs_output_dir,
        output_prefix=prefix,
        max_frames=max_frames,
    )
    _run_extractor(args)

    landmark_csv = srs_output_dir / f"{prefix}_pose_landmarks_stabilized.csv"
    frame_quality_csv = srs_output_dir / f"{prefix}_frame_quality.csv"
    stable_pose_video = srs_output_dir / f"{prefix}_stable_pose.mp4"
    quality_boxes_video = srs_output_dir / f"{prefix}_stable_pose_quality_boxes.mp4"
    comparison_video = srs_output_dir / f"{prefix}_raw_vs_stable.mp4"

    frame_records = _write_project_frames(
        source,
        config=config,
        clip_id=resolved_clip_id,
        condition_id=condition_id,
        max_frames=max_frames,
    )
    pose_records = _read_srs_landmark_csv(
        landmark_csv,
        clip_id=resolved_clip_id,
        condition_id=condition_id,
    )
    project_pose_csv = pose_path(config.data_dir, resolved_clip_id, condition_id)
    write_pose_records(project_pose_csv, pose_records)
    frame_manifest_csv = frame_manifest_path(config.data_dir, resolved_clip_id, condition_id)
    write_frame_records(frame_manifest_csv, frame_records)

    return SRS2DResult(
        clip_id=resolved_clip_id,
        condition_id=condition_id,
        output_dir=srs_output_dir,
        landmark_csv=landmark_csv,
        frame_quality_csv=frame_quality_csv,
        stable_pose_video=stable_pose_video,
        quality_boxes_video=quality_boxes_video,
        comparison_video=comparison_video,
        project_pose_csv=project_pose_csv,
        frame_manifest_csv=frame_manifest_csv,
        frame_count=len(frame_records),
    )


def run_video_2d_3d_pipeline(
    input_video: str | Path,
    *,
    config: RuntimeConfig,
    clip_id: str | None = None,
    condition_id: str = "srs_2d_pose",
    output_prefix: str | None = None,
    max_frames: int = 0,
    smooth: bool = True,
    render_overlay: bool = True,
) -> TwoDThreeDResult:
    """Run SRS 2D, import matching GVHMR 3D output, and render 3D overlay."""

    srs2d = run_srs_2d_pipeline(
        input_video,
        config=config,
        clip_id=clip_id,
        condition_id=condition_id,
        output_prefix=output_prefix,
        max_frames=max_frames,
    )
    clip = _clip_metadata_for_video(input_video, srs2d.clip_id, config)
    plan = build_pose3d_plan(clip, config, input_condition_id=condition_id)
    lift_pose_sequence(plan, clip, config)

    smoothed_csv = None
    overlay_video = None
    overlay_condition = plan.output_condition_id
    if smooth:
        smooth_results = smooth_pose3d_files([srs2d.clip_id], config, [plan.output_condition_id])
        if smooth_results:
            smoothed_csv = smooth_results[0].pose3d_csv
            overlay_condition = smooth_results[0].condition_id
    if render_overlay:
        overlay_results = render_pose3d_overlays([srs2d.clip_id], config, [overlay_condition])
        if overlay_results:
            overlay_video = overlay_results[0].overlay_video

    return TwoDThreeDResult(
        srs2d=srs2d,
        pose3d_csv=plan.output_pose3d_path,
        smoothed_pose3d_csv=smoothed_csv,
        overlay3d_video=overlay_video,
    )


def _extractor_args(
    *,
    input_video: Path,
    output_dir: Path,
    output_prefix: str,
    max_frames: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_video,
        output_dir=output_dir,
        output_prefix=output_prefix,
        max_frames=max_frames,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.55,
        model_complexity=2,
        visibility_threshold=extractor.VISIBILITY_THRESHOLD,
        quality_fast_limb_visibility=extractor.QUALITY_FAST_LIMB_VISIBILITY,
        quality_core_visibility=extractor.QUALITY_CORE_VISIBILITY,
        quality_other_visibility=extractor.QUALITY_OTHER_VISIBILITY,
        quality_aux_visibility=extractor.QUALITY_AUX_VISIBILITY,
        static_fast_limb_visibility=extractor.STATIC_FAST_LIMB_VISIBILITY,
        static_other_visibility=extractor.STATIC_OTHER_VISIBILITY,
        fusion_fast_limb_distance=extractor.FUSION_FAST_LIMB_DISTANCE,
        fusion_other_distance=extractor.FUSION_OTHER_DISTANCE,
        fusion_visibility_margin=extractor.FUSION_VISIBILITY_MARGIN,
        person_center_jump_abs=extractor.PERSON_CENTER_JUMP_ABS,
        person_center_jump_scale=extractor.PERSON_CENTER_JUMP_SCALE,
        person_scale_ratio_min=extractor.PERSON_SCALE_RATIO_MIN,
        person_scale_ratio_max=extractor.PERSON_SCALE_RATIO_MAX,
        body_jump_abs=extractor.BODY_JUMP_ABS,
        body_jump_scale=extractor.BODY_JUMP_SCALE,
        core_smooth_seconds=extractor.CORE_SMOOTH_SECONDS,
        limb_smooth_seconds=extractor.LIMB_SMOOTH_SECONDS,
        fast_limb_smooth_seconds=extractor.FAST_LIMB_SMOOTH_SECONDS,
    )


def _run_extractor(args: SimpleNamespace) -> None:
    previous_prefix = extractor.OUTPUT_PREFIX
    try:
        extractor.OUTPUT_PREFIX = args.output_prefix
        extractor.apply_runtime_config(args)
        baseline_data = extractor.detect_video(args.input, args, static_image_mode=True, smooth_landmarks=False)
        tracking_data = extractor.detect_video(args.input, args, static_image_mode=False, smooth_landmarks=True)
        data, fusion_replacements = extractor.fuse_static_and_tracking(baseline_data, tracking_data)
        accepted, reason, _centers, scales = extractor.reject_bad_frames(
            data["raw_xyz"],
            data["raw_visibility"],
            data["detected"],
        )
        stable_xyz, stable_visibility, smoothing = extractor.stabilize_landmarks(
            data["raw_xyz"],
            data["raw_visibility"],
            accepted,
            data["fps"],
        )
        frame_quality = extractor.classify_all_frames(data)
        output_data = dict(data)
        output_data["raw_xyz"] = baseline_data["raw_xyz"]
        output_data["raw_visibility"] = baseline_data["raw_visibility"]
        output_data["detected"] = baseline_data["detected"]

        raw_path, stable_path, quality_path, compare_path = extractor.write_videos(
            output_data,
            stable_xyz,
            stable_visibility,
            frame_quality,
            args.output_dir,
        )
        landmark_csv = extractor.write_csv(output_data, accepted, reason, stable_xyz, stable_visibility, args.output_dir)
        quality_csv = extractor.write_frame_quality_csv(data, frame_quality, args.output_dir)
        angle_csv = extractor.write_angle_csv(data, stable_xyz, args.output_dir)
        contact_sheet = extractor.write_contact_sheet(output_data, stable_xyz, stable_visibility, args.output_dir)
        quality_contact_sheet = extractor.write_quality_contact_sheet(
            output_data,
            stable_xyz,
            stable_visibility,
            frame_quality,
            args.output_dir,
        )
        _write_extractor_summary(
            args=args,
            data=data,
            baseline_data=baseline_data,
            tracking_data=tracking_data,
            accepted=accepted,
            reason=reason,
            scales=scales,
            stable_xyz=stable_xyz,
            stable_visibility=stable_visibility,
            frame_quality=frame_quality,
            fusion_replacements=fusion_replacements,
            smoothing=smoothing,
            outputs={
                "raw_video": str(raw_path),
                "stable_video": str(stable_path),
                "quality_boxes_video": str(quality_path),
                "comparison_video": str(compare_path),
                "landmark_csv": str(landmark_csv),
                "frame_quality_csv": str(quality_csv),
                "angle_csv": str(angle_csv),
                "contact_sheet": str(contact_sheet),
                "quality_contact_sheet": str(quality_contact_sheet),
            },
        )
    finally:
        extractor.OUTPUT_PREFIX = previous_prefix


def _write_project_frames(
    input_video: Path,
    *,
    config: RuntimeConfig,
    clip_id: str,
    condition_id: str,
    max_frames: int,
) -> list[FrameRecord]:
    cv2 = extractor.cv2
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or config.target_fps)
    output_dir = frame_dir(config.data_dir, clip_id, condition_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames and frame_index >= max_frames:
                break
            frame_path = output_dir / f"{clip_id}__{condition_id}__frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write frame: {frame_path}")
            height, width = frame.shape[:2]
            records.append(
                FrameRecord(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    frame_path=frame_path,
                    width=width,
                    height=height,
                )
            )
            frame_index += 1
    finally:
        capture.release()
    return records


def _read_srs_landmark_csv(
    path: Path,
    *,
    clip_id: str,
    condition_id: str,
) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stable_x = _optional_float(row.get("stable_x_px"))
            stable_y = _optional_float(row.get("stable_y_px"))
            visibility = _optional_float(row.get("stable_visibility"))
            records.append(
                PoseRecord(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    frame_index=int(row["frame"]),
                    timestamp_sec=float(row["time_sec"]),
                    joint_name=str(row["landmark"]),
                    x=stable_x,
                    y=stable_y,
                    visibility=visibility,
                    confidence=visibility,
                    backend="srs_2d_pose_jiaming",
                    inference_time_ms=None,
                )
            )
    return records


def _write_extractor_summary(**kwargs) -> None:
    import json
    import numpy as np

    args = kwargs["args"]
    data = kwargs["data"]
    baseline_data = kwargs["baseline_data"]
    tracking_data = kwargs["tracking_data"]
    accepted = kwargs["accepted"]
    reason = kwargs["reason"]
    scales = kwargs["scales"]
    stable_xyz = kwargs["stable_xyz"]
    stable_visibility = kwargs["stable_visibility"]
    frame_quality = kwargs["frame_quality"]
    outputs = kwargs["outputs"]

    accepted_scales = scales.copy()
    if np.isfinite(accepted_scales).sum() == 0:
        accepted_scales[:] = 1.0
    else:
        accepted_scales = extractor.interpolate_1d(accepted_scales, np.isfinite(accepted_scales))
    raw_jitter = extractor.normalized_jitter(
        baseline_data["raw_xyz"],
        baseline_data["raw_visibility"],
        baseline_data["detected"],
        accepted_scales,
    )
    tracking_jitter = extractor.normalized_jitter(
        tracking_data["raw_xyz"],
        tracking_data["raw_visibility"],
        tracking_data["detected"],
        accepted_scales,
    )
    fused_jitter = extractor.normalized_jitter(
        data["raw_xyz"],
        data["raw_visibility"],
        data["detected"],
        accepted_scales,
    )
    stable_jitter = extractor.normalized_jitter(
        stable_xyz,
        stable_visibility,
        np.ones(len(data["frames"]), dtype=bool),
        accepted_scales,
    )
    reduction = None
    if raw_jitter and stable_jitter and raw_jitter["median"] > 0:
        reduction = round((1.0 - stable_jitter["median"] / raw_jitter["median"]) * 100.0, 1)

    summary_json = extractor.output_path(args.output_dir, "stabilization_summary.json")
    metrics = {
        "input_video": str(args.input),
        "output_prefix": args.output_prefix,
        "width": data["width"],
        "height": data["height"],
        "fps": data["fps"],
        "processed_frames": len(data["frames"]),
        "detected_frames": int(baseline_data["detected"].sum()),
        "tracking_detected_frames": int(tracking_data["detected"].sum()),
        "accepted_frames": int(accepted.sum()),
        "green_2d_direct_frames": int(sum(1 for item in frame_quality if item["quality_label"] == "green")),
        "yellow_2d_3d_fusion_candidate_frames": int(sum(1 for item in frame_quality if item["quality_label"] == "yellow")),
        "red_unusable_frames": int(sum(1 for item in frame_quality if item["quality_label"] == "red")),
        "usable_full_body_frames": int(sum(1 for item in frame_quality if item["quality_label"] == "green")),
        "fusion_candidate_frames": int(sum(1 for item in frame_quality if item["fusion_candidate"])),
        "occluded_or_incomplete_frames": int(sum(1 for item in frame_quality if item["quality_label"] == "red")),
        "usable_full_body_rate": round(float(sum(1 for item in frame_quality if item["quality_label"] == "green") / len(frame_quality)), 4),
        "fusion_candidate_rate": round(float(sum(1 for item in frame_quality if item["fusion_candidate"]) / len(frame_quality)), 4),
        "temporal_bridge_frames": int(sum(1 for item in frame_quality if item["temporal_status"] == "temporal_bridge")),
        "short_gap_interpolated_frames": int(sum(1 for item in frame_quality if item["temporal_status"] == "short_gap_interpolated")),
        "swing_phase_occlusion_candidate_frames": int(sum(1 for item in frame_quality if item["temporal_status"] == "swing_phase_occlusion_candidate")),
        "persistent_occlusion_frames": int(sum(1 for item in frame_quality if item["temporal_status"] == "persistent_occlusion")),
        "fusion_replacements": int(kwargs["fusion_replacements"]),
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "model_complexity": args.model_complexity,
        "raw_jitter": raw_jitter,
        "tracking_jitter": tracking_jitter,
        "fused_jitter": fused_jitter,
        "stable_jitter": stable_jitter,
        "median_jitter_reduction_percent": reduction,
        "frame_status_counts": {item: reason.count(item) for item in set(reason)},
        "outputs": outputs,
        **kwargs["smoothing"],
    }
    report_path, rejected_counts = extractor.write_report(
        args.input,
        data,
        accepted,
        reason,
        metrics,
        outputs | {"summary_json": str(summary_json)},
        args.output_dir,
    )
    metrics["frame_status_counts"] = rejected_counts
    metrics["outputs"] = outputs | {"summary_json": str(summary_json), "report": str(report_path)}
    summary_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip_metadata_for_video(input_video: str | Path, clip_id: str, config: RuntimeConfig):
    from baseball_pose.io.metadata import ClipMetadata

    return ClipMetadata(
        clip_id=clip_id,
        source_path=Path(input_video),
        action_type="batting",
        fps_target=config.target_fps,
        difficulty_tags=(),
    )


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _safe_clip_id(value: str) -> str:
    normalized = []
    for char in value.lower():
        if char.isalnum():
            normalized.append(char)
        elif normalized and normalized[-1] != "_":
            normalized.append("_")
    return "".join(normalized).strip("_") or "clip"
