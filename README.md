# Baseball Analysis Model

This repository packages the model-side baseball pose pipeline:

```text
single input video
  -> SRS 2D stabilized pose extraction
  -> project 2D pose/frame CSV contract
  -> GVHMR/external video-HMR 3D import
  -> 3D smoothing
  -> overlays3d video output
```

The SRS 2D pose extractor under `src/baseball_pose/srs2d/` is Jiaming's work.
Its original README and model card are preserved in
`docs/srs_2d_pose_model_package/`.

Generated data, raw videos, model weights, and large external model repositories
are intentionally excluded by `.gitignore`.

## Included Pipeline

Key files:

- `src/baseball_pose/srs2d/extractor.py`: Jiaming's SRS 2D stabilized pose extractor.
- `src/baseball_pose/srs2d/pipeline.py`: adapter that runs SRS 2D, exports the project pose CSV, and writes frame manifests.
- `scripts/build_gvhmr_input_from_frames.py`: build a GVHMR-safe input video from sampled frames.
- `scripts/export_gvhmr_joints.py`: convert GVHMR `hmr4d_results.pt` outputs to the flat project 3D CSV contract.
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

## GVHMR/SMPL Assets

Do not vendor the full GVHMR source tree or model weights into this repository.
The 3D stage consumes either:

- a ready-made CSV at `data/external_pose3d/gvhmr/{clip_id}.csv`, or
- a GVHMR `hmr4d_results.pt` converted by `scripts/export_gvhmr_joints.py` from
  an existing GVHMR runtime environment.

Download the required model/body-model files from the official upstream
sources:

- GVHMR official code and install notes: <https://github.com/zju3dv/GVHMR>
- GVHMR pretrained checkpoint folder from the official install doc:
  <https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link>
- SMPL official download page: <https://smpl.is.tue.mpg.de/>
- SMPL-X official download page: <https://smpl-x.is.tue.mpg.de/>

SMPL and SMPL-X require registration and license agreement before download. Do
not commit these files to this repo.

Place the downloaded assets in the checkpoint tree expected by your GVHMR
runtime. A typical local layout is:

```text
{GVHMR_ROOT}/inputs/checkpoints/
├── body_models/
│   ├── smpl/
│   │   └── SMPL_NEUTRAL.pkl        # or SMPL_{GENDER}.pkl from SMPL
│   └── smplx/
│       └── SMPLX_NEUTRAL.npz       # or SMPLX_{GENDER}.npz from SMPL-X
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
├── yolo/
│   └── yolov8x.pt
└── dpvo/
    └── dpvo.pth                    # optional when using DPVO instead of SimpleVO
```

For this repo's CSV conversion step, `export_gvhmr_joints.py` needs access to
the GVHMR Python package utilities and body-model helper files from an existing
GVHMR environment. Point `--gvhmr-root` at that environment; it does not have to
live inside this repository.

After GVHMR produces `hmr4d_results.pt`, convert it to this repo's 3D CSV
contract:

```bash
GVHMR_ROOT=/path/to/GVHMR
python scripts/export_gvhmr_joints.py \
  --gvhmr-root "$GVHMR_ROOT" \
  --result outputs/gvhmr_runs/clip_a/hmr4d_results.pt \
  --output data/external_pose3d/gvhmr/clip_a.csv \
  --clip-id clip_a
```

## Output Layout

For clip `clip_a` and condition `srs_2d_pose`, the integrated entry writes:

```text
outputs/srs2d/clip_a/clip_a_stable_pose.mp4
outputs/srs2d/clip_a/clip_a_stable_pose_quality_boxes.mp4
outputs/srs2d/clip_a/clip_a_raw_vs_stable.mp4
data/interim/frames/clip_a/srs_2d_pose.csv
data/processed/poses/clip_a/srs_2d_pose.csv
data/processed/poses3d/clip_a/srs_2d_pose_3d.csv
data/processed/poses3d/clip_a/srs_2d_pose_3d_smooth.csv
outputs/overlays3d/clip_a__srs_2d_pose_3d_smooth.mp4
```

The 3D overlay naming matches the existing `overlays3d` format:

```text
{clip_id}__{condition_id}_3d_smooth.mp4
```

## GVHMR Data Contract

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

## Commands

Run only Jiaming's SRS 2D extractor and export project-compatible 2D CSVs:

```bash
baseball-pose --config configs/default.yaml run-srs-2d \
  --input path/to/video.mp4 \
  --clip-id clip_a
```

Run the merged 2D+3D entry for one video after the matching GVHMR CSV exists:

```bash
baseball-pose --config configs/default.yaml run-video-2d-3d \
  --input path/to/video.mp4 \
  --clip-id clip_a
```

3D-only commands are still available:

```bash
baseball-pose --config configs/default.yaml validate-config
baseball-pose --config configs/default.yaml plan-3d
baseball-pose --config configs/default.yaml lift-pose-3d
baseball-pose --config configs/default.yaml smooth-pose-3d
baseball-pose --config configs/default.yaml render-overlays-3d
```
