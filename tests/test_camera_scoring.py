import numpy as np

from gaussiangpt_ae.data.ase_camera import ASECameras
from gaussiangpt_ae.data.camera_scoring import score_cameras_for_chunk


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
    assert "image_coverage" in scores[0]
    assert "visible_ratio" in scores[0]
    assert "valid_projection" in scores[0]
    assert scores[0]["valid_projection"] is True
    assert scores[0]["image_coverage"] > 0.0
