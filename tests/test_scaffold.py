from gaussiangpt_ae import __version__
from gaussiangpt_ae.spec import FEATURE_DIM, FEATURE_ORDER


def test_package_imports() -> None:
    assert __version__


def test_feature_spec_matches_project_contract() -> None:
    assert FEATURE_ORDER == ("relative_xyz", "color", "opacity", "scale", "rotation")
    assert FEATURE_DIM == 14
