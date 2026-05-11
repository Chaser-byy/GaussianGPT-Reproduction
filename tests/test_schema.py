import numpy as np
import pytest

from gaussiangpt_ae.data.schema import GaussianScene, validate_gaussian_scene


def make_scene() -> GaussianScene:
    return GaussianScene(
        scene_id="scene_0001",
        xyz=np.zeros((2, 3), dtype=np.float32),
        color=np.zeros((2, 3), dtype=np.float32),
        opacity=np.zeros((2, 1), dtype=np.float32),
        scale=np.zeros((2, 3), dtype=np.float32),
        rotation=np.zeros((2, 4), dtype=np.float32),
        metadata={},
    )


def test_validate_gaussian_scene_passes() -> None:
    validate_gaussian_scene(make_scene())


def test_validate_gaussian_scene_rejects_bad_shape() -> None:
    scene = make_scene()
    scene.opacity = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="opacity shape"):
        validate_gaussian_scene(scene)

