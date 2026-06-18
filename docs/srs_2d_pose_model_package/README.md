# SRS 2D Pose Skeleton Extractor

这是当前小组选择使用的 2D 骨架提取模型包。核心脚本是 `srs_2d_pose_extractor.py`，底层使用 MediaPipe Pose，外层加入了视频追踪、静态/追踪融合、坏帧剔除、缺失点插值、时序平滑和绿/黄/红质量判断。

## 环境安装

建议使用 Python 3.10-3.12。

```powershell
pip install -r requirements.txt
```

## 运行命令

```powershell
python srs_2d_pose_extractor.py --input "C:\path\to\video.mp4" --output-dir ".\outputs"
```

可选：自定义输出文件名前缀。

```powershell
python srs_2d_pose_extractor.py --input "C:\path\to\video.mp4" --output-dir ".\outputs" --output-prefix "player01_hit"
```

快速测试前 100 帧：

```powershell
python srs_2d_pose_extractor.py --input "C:\path\to\video.mp4" --output-dir ".\outputs" --max-frames 100
```

## 当前默认参数

默认使用当前项目里效果最好的 `sensitive_far_lowlight` 风格参数：

- `model_complexity=2`
- `min_detection_confidence=0.45`
- `min_tracking_confidence=0.55`
- `visibility_threshold=0.18`
- 核心点平滑约 `0.10s`
- 普通肢体点平滑约 `0.065s`
- 手腕/脚踝等快速点平滑约 `0.035s`

这组参数更适合棒球视频里的远景、低光、快速挥棒/投球场景。

## 输出文件

假设输入视频名是 `hit01.mp4`，输出目录会生成：

- `hit01_old_static_pose.mp4`：旧静态逐帧检测骨架视频，用作 baseline。
- `hit01_stable_pose.mp4`：稳定后的 2D 骨架视频。
- `hit01_stable_pose_quality_boxes.mp4`：带绿/黄/红质量框的视频。
- `hit01_raw_vs_stable.mp4`：旧方法和稳定方法左右对比视频。
- `hit01_pose_landmarks_stabilized.csv`：每帧 33 个关键点坐标与 visibility。
- `hit01_frame_quality.csv`：每帧质量标签、可用性、遮挡原因。
- `hit01_stable_angles.csv`：稳定后左右肘、左右膝角度。
- `hit01_raw_vs_stable_contact_sheet.jpg`：抽帧对比图。
- `hit01_quality_boxes_contact_sheet.jpg`：质量判断抽帧图。
- `hit01_stabilization_summary.json`：整体统计结果。
- `hit01_stabilization_notes.md`：自动生成的解释报告。

## 质量标签含义

- `green`：关键身体点完整，适合直接做 2D 动作分析。
- `yellow`：轻微遮挡或局部不稳定，可保留给后续 2D+3D/Vicon 融合。
- `red`：遮挡严重或结构不可靠，不建议直接用于精细角度分析。

## 给同学整合时的统一接口

其他模型如果要接进来，建议也输出同样的 CSV 字段：

- `frame`
- `time_sec`
- `landmark`
- `stable_x_px`
- `stable_y_px`
- `stable_visibility`
- `quality_label`

这样后续的动作指标、可视化和模型 benchmark 可以复用同一套流程。

