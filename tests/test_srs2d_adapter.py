from pathlib import Path

from baseball_pose.srs2d.pipeline import _read_srs_landmark_csv


def test_srs_landmark_csv_converts_to_project_pose_records(tmp_path):
    source = tmp_path / "clip_pose_landmarks_stabilized.csv"
    source.write_text(
        "\n".join(
            [
                "frame,time_sec,detected,accepted_for_stabilization,frame_status,landmark,raw_x_px,raw_y_px,raw_visibility,stable_x_px,stable_y_px,stable_visibility",
                "0,0.0,1,1,accepted,left_wrist,10,20,0.7,12.5,22.5,0.8",
            ]
        ),
        encoding="utf-8",
    )

    records = _read_srs_landmark_csv(
        Path(source),
        clip_id="clip_a",
        condition_id="srs_2d_pose",
    )

    assert len(records) == 1
    assert records[0].clip_id == "clip_a"
    assert records[0].condition_id == "srs_2d_pose"
    assert records[0].joint_name == "left_wrist"
    assert records[0].x == 12.5
    assert records[0].y == 22.5
    assert records[0].visibility == 0.8
    assert records[0].backend == "srs_2d_pose_jiaming"
