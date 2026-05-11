"""Shared Gaussian primitive feature specification."""

GAUSSIAN_FIELDS = {
    "xyz": 3,
    "color": 3,
    "opacity": 1,
    "scale": 3,
    "rotation": 4,
}

FEATURE_ORDER = (
    "relative_xyz",
    "color",
    "opacity",
    "scale",
    "rotation",
)

FEATURE_DIMS = {
    "relative_xyz": 3,
    "color": 3,
    "opacity": 1,
    "scale": 3,
    "rotation": 4,
}

FEATURE_DIM = sum(FEATURE_DIMS.values())
