import numpy as np

from gaussiangpt_ae.data.ase import ASECameras
from gaussiangpt_ae.data.sampler import score_cameras_for_chunk, select_cameras_for_chunk


def test_score_cameras_for_chunk_returns_scores() -> None:
    cameras = ASECameras(
        scene_id="00000",
        width=640,
        height=480,
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=240.0,
        share_intrinsics=True,
        zipped=False,
        crop_edge=0,
        frames_num=1,
        frames=[
            {
                "camera_id": "frame0000000",
                "file_path": "rgb_undistorted/frame0000000.jpg",
                "transform_matrix": np.eye(4, dtype=np.float32),
            }
        ],
    )

    scores = score_cameras_for_chunk(
        cameras,
        chunk_world_min=np.asarray([-0.5, -0.5, 2.0], dtype=np.float32),
        chunk_world_max=np.asarray([0.5, 0.5, 3.0], dtype=np.float32),
        top_k=12,
    )

    assert len(scores) == 1
    assert scores[0]["camera_id"] == "frame0000000"
    assert "chunk_coverage" in scores[0]
    assert "image_coverage" in scores[0]
    assert "visible_ratio" in scores[0]
    assert "visible_corners" in scores[0]
    assert "total_corners" in scores[0]
    assert "projected_bbox" in scores[0]
    assert "projected_bbox_area" in scores[0]
    assert "intersection_area" in scores[0]
    assert "depth_min" in scores[0]
    assert "depth_max" in scores[0]
    assert "near_plane_crossing" in scores[0]
    assert "valid_projection" in scores[0]
    assert scores[0]["valid_projection"] is True
    assert scores[0]["chunk_coverage"] > 0.0
    assert scores[0]["image_coverage"] > 0.0
    assert scores[0]["visible_ratio"] == scores[0]["visible_corners"] / 8


def test_select_cameras_prefers_chunk_coverage_not_image_coverage() -> None:
    cameras = ASECameras(
        scene_id="00000",
        width=640,
        height=480,
        fx=100.0,
        fy=100.0,
        cx=320.0,
        cy=240.0,
        share_intrinsics=True,
        zipped=False,
        crop_edge=0,
        frames_num=1,
        frames=[
            {
                "camera_id": "frame0000000",
                "frame_index": 0,
                "frame_id": 0,
                "file_path": "rgb_undistorted/frame0000000.jpg",
                "transform_matrix": np.eye(4, dtype=np.float32),
            }
        ],
    )

    selected = select_cameras_for_chunk(
        cameras,
        chunk_world_min=np.asarray([-0.5, -0.5, 2.0], dtype=np.float32),
        chunk_world_max=np.asarray([0.5, 0.5, 3.0], dtype=np.float32),
        top_k=1,
    )

    assert len(selected) == 1
    assert selected[0]["selection_mode"] == "preferred"
    assert selected[0]["chunk_coverage"] >= 0.4
    assert selected[0]["image_coverage"] < 0.4
