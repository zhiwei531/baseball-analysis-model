# GVHMR 3D Pipeline Notes

This repository keeps the model-side 3D pipeline separate from upstream video
sampling and 2D pose generation.

## Contract

GVHMR or another video-HMR backend runs externally and produces one CSV or NPZ
file per clip. The project adapter imports that output into a flat 3D joint CSV.

Required CSV columns:

```text
frame_index,joint_name,x_3d,y_3d,z_3d
```

Optional CSV columns:

```text
clip_id,timestamp_sec,confidence,score,scale_mode,lift_backend
```

## Stages

```text
external GVHMR result
  -> lift-pose-3d
  -> smooth-pose-3d
  -> render-overlays-3d
```

`lift-pose-3d` uses a sampled frame manifest when available so timestamps line
up with the upstream pipeline. If the frame manifest is missing, it can use an
existing 2D pose CSV as a timing source and optional confidence prior. This repo
does not run 2D detection.

## GVHMR Export

Build a video from upstream sampled frames when raw video rotation metadata is
unsafe:

```bash
python scripts/build_gvhmr_input_from_frames.py \
  --frames-csv data/interim/frames/clip_a/gvhmr_input.csv \
  --output outputs/gvhmr_inputs/clip_a.mp4 \
  --fps 30
```

After running GVHMR externally, export its result to the project CSV contract:

```bash
python scripts/export_gvhmr_joints.py \
  --gvhmr-root external/GVHMR \
  --result outputs/gvhmr_runs/clip_a/hmr4d_results.pt \
  --output data/external_pose3d/gvhmr/clip_a.csv \
  --clip-id clip_a
```

Then import and smooth:

```bash
baseball-pose --config configs/default.yaml lift-pose-3d
baseball-pose --config configs/default.yaml smooth-pose-3d
```
