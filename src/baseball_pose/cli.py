"""Command-line entry points for the GVHMR 3D baseball pipeline."""

from __future__ import annotations

import argparse

from baseball_pose.config import load_config
from baseball_pose.io.metadata import load_clips
from baseball_pose.pipeline.overlays3d import render_pose3d_overlays
from baseball_pose.pipeline.pose3d import build_pose3d_plan, lift_pose_sequence
from baseball_pose.pipeline.postprocess3d import smooth_pose3d_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baseball-pose")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "command",
        choices=[
            "validate-config",
            "plan-3d",
            "lift-pose-3d",
            "smooth-pose-3d",
            "render-overlays-3d",
            "run-srs-2d",
            "run-video-2d-3d",
        ],
        help="Command to run.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        help="Clip id to process. Repeat to process multiple clips. Defaults to all configured clips.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help="Input or 3D condition id. Defaults to config experiment conditions.",
    )
    parser.add_argument(
        "--input",
        help="Input video path for run-srs-2d or run-video-2d-3d.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix for the SRS 2D extractor. Defaults to clip id.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick debugging. 0 means all frames.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "validate-config":
        print(f"Loaded config: {config.path}")
        print(f"Configured clips: {len(config.clip_ids)}")
        print(f"Configured 2D input conditions: {', '.join(config.condition_ids)}")
        print(f"Configured 3D conditions: {', '.join(config.pose3d_condition_ids)}")
        return

    if args.command == "plan-3d":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        input_conditions = args.condition or config.condition_ids
        for clip in clips:
            if clip.clip_id not in clip_filter:
                continue
            for condition_id in input_conditions:
                plan = build_pose3d_plan(
                    clip,
                    config,
                    input_condition_id=condition_id,
                )
                print(f"{plan.clip_id}: {plan.input_condition_id} -> {plan.output_condition_id}")
                print(f"  source condition: {plan.source_condition_id}")
                print(f"  backend: {plan.backend}")
                print(f"  root joint: {plan.root_joint}")
                print(f"  input pose: {plan.input_pose_path}")
                print(f"  input frames: {plan.input_frames_path}")
                print(f"  output pose3d: {plan.output_pose3d_path}")
                print(f"  output feature3d: {plan.output_feature3d_path}")
        return

    if args.command == "lift-pose-3d":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        input_conditions = args.condition or config.condition_ids
        for clip in clips:
            if clip.clip_id not in clip_filter:
                continue
            for condition_id in input_conditions:
                plan = build_pose3d_plan(
                    clip,
                    config,
                    input_condition_id=condition_id,
                )
                print(f"Planned 3D lift: {plan.clip_id} {plan.input_condition_id} -> {plan.output_condition_id}")
                record_count = lift_pose_sequence(plan, clip, config)
                print(f"  source condition: {plan.source_condition_id}")
                print(f"  output pose3d: {plan.output_pose3d_path}")
                print(f"  wrote {record_count} 3D joint records")
        return

    if args.command == "smooth-pose-3d":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        condition_ids = args.condition or config.pose3d_condition_ids
        if not condition_ids:
            raise ValueError(
                "smooth-pose-3d requires at least one --condition or experiments.default_3d_conditions."
            )
        results = smooth_pose3d_files(clip_ids, config, condition_ids)
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: {result.pose_record_count} 3D records")
            print(f"  source: {result.source_condition_id}")
            print(f"  pose3d: {result.pose3d_csv}")
        return

    if args.command == "render-overlays-3d":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        condition_ids = args.condition or config.pose3d_condition_ids
        if not condition_ids:
            raise ValueError(
                "render-overlays-3d requires at least one --condition or experiments.default_3d_conditions."
            )
        results = render_pose3d_overlays(clip_ids, config, condition_ids)
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: {result.frame_count} frames")
            print(f"  overlay3d: {result.overlay_video}")
        return

    if args.command == "run-srs-2d":
        from baseball_pose.srs2d.pipeline import run_srs_2d_pipeline

        if not args.input:
            raise ValueError("run-srs-2d requires --input.")
        clip_id = args.clip_id[0] if args.clip_id else None
        condition_id = args.condition[0] if args.condition else "srs_2d_pose"
        result = run_srs_2d_pipeline(
            args.input,
            config=config,
            clip_id=clip_id,
            condition_id=condition_id,
            output_prefix=args.output_prefix,
            max_frames=args.max_frames,
        )
        print(f"{result.clip_id}/{result.condition_id}: {result.frame_count} frames")
        print(f"  srs2d stable video: {result.stable_pose_video}")
        print(f"  srs2d quality video: {result.quality_boxes_video}")
        print(f"  project pose csv: {result.project_pose_csv}")
        print(f"  frames csv: {result.frame_manifest_csv}")
        return

    if args.command == "run-video-2d-3d":
        from baseball_pose.srs2d.pipeline import run_video_2d_3d_pipeline

        if not args.input:
            raise ValueError("run-video-2d-3d requires --input.")
        clip_id = args.clip_id[0] if args.clip_id else None
        condition_id = args.condition[0] if args.condition else "srs_2d_pose"
        result = run_video_2d_3d_pipeline(
            args.input,
            config=config,
            clip_id=clip_id,
            condition_id=condition_id,
            output_prefix=args.output_prefix,
            max_frames=args.max_frames,
        )
        print(f"{result.srs2d.clip_id}/{result.srs2d.condition_id}: {result.srs2d.frame_count} frames")
        print(f"  srs2d stable video: {result.srs2d.stable_pose_video}")
        print(f"  project pose csv: {result.srs2d.project_pose_csv}")
        print(f"  pose3d: {result.pose3d_csv}")
        print(f"  pose3d smooth: {result.smoothed_pose3d_csv}")
        print(f"  overlay3d: {result.overlay3d_video}")
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
