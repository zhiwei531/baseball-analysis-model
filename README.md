# Baseball Analysis Model

This repository contains only the GVHMR-oriented 3D pipeline extracted from the
baseball analysis project. It does not include the 2D detector pipeline,
RTMPose/MediaPipe detector implementations, YOLO/object tracking, reports, raw
videos, or generated outputs.

## Included Pipeline

The supported flow is:

```text
GVHMR/external video-HMR result
  -> lift-pose-3d
  -> smooth-pose-3d
  -> render-overlays-3d
```

Key files:

- `scripts/build_gvhmr_input_from_frames.py`: build a GVHMR-safe input video
  from frames sampled by an upstream pipeline.
- `scripts/export_gvhmr_joints.py`: convert GVHMR `hmr4d_results.pt` outputs
  to the flat project 3D CSV contract.
- `src/baseball_pose/pose3d/`: 3D pose schemas and external HMR import adapter.
- `src/baseball_pose/pipeline/pose3d.py`: GVHMR/external-HMR import stage.
- `src/baseball_pose/postprocess/smoothing3d.py`: temporal 3D smoothing.
- `src/baseball_pose/pipeline/overlays3d.py`: 3D overlay video rendering.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Data Contract

External GVHMR CSV files are expected at:

```text
{data_dir}/external_pose3d/{backend}/{clip_id}.csv
```

Required columns:

```text
frame_index,joint_name,x_3d,y_3d,z_3d
```

Optional columns:

```text
clip_id,timestamp_sec,confidence,score,scale_mode,lift_backend
```

The imported 3D pose CSV is written to:

```text
{data_dir}/processed/poses3d/{clip_id}/{condition}_3d.csv
```

## Commands

```bash
baseball-pose --config configs/experiments/gvhmr_benchmark_baseball_1.yaml validate-config
baseball-pose --config configs/experiments/gvhmr_benchmark_baseball_1.yaml plan-3d
baseball-pose --config configs/experiments/gvhmr_benchmark_baseball_1.yaml lift-pose-3d
baseball-pose --config configs/experiments/gvhmr_benchmark_baseball_1.yaml smooth-pose-3d
baseball-pose --config configs/experiments/gvhmr_benchmark_baseball_1.yaml render-overlays-3d
```

`lift-pose-3d` can use existing sampled frame manifests or an upstream 2D pose
CSV only as timing and optional gating inputs. This repository intentionally
does not run 2D pose detection.
