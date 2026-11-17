import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


def generate_signal_image(
    n_folds,
    angles_deg,
    amplitudes,
    imshape=(64, 64),
    radius=20,
    sigma=1.5,
):
    """Generate an image of n-fold signals.

    Each signal is drawn as `n_folds` Gaussian peaks spaced evenly on a ring
    around the image center, with the first peak along the signal's
    orientation angle. Angles are measured from the +column axis toward the
    +row axis, so the true angle of every signal is known exactly.

    Parameters
    ----------
    n_folds : int
        The number of peaks in each signal.
    angles_deg : list of float
        The orientation angle of each signal, in degrees.
    amplitudes : list of float
        The peak height of each signal.
    imshape : tuple of int, optional
        The shape of the image as (rows, columns). Default is (64, 64).
    radius : float, optional
        The distance of the peaks from the image center, in pixels. Default
        is 20.
    sigma : float, optional
        The standard deviation of each Gaussian peak, in pixels. Default is
        1.5.

    Returns
    -------
    image : ndarray
        The image, of shape (rows, columns).
    """
    row_grid, col_grid = np.mgrid[0 : imshape[0], 0 : imshape[1]]
    center_row, center_col = imshape[0] // 2, imshape[1] // 2
    image = np.zeros(imshape)
    for signal_angle, amplitude in zip(angles_deg, amplitudes):
        for peak_index in range(n_folds):
            peak_angle = np.deg2rad(signal_angle + peak_index * 360 / n_folds)
            peak_row = center_row + radius * np.sin(peak_angle)
            peak_col = center_col + radius * np.cos(peak_angle)
            squared_distance = (row_grid - peak_row) ** 2 + (
                col_grid - peak_col
            ) ** 2
            image += amplitude * np.exp(-squared_distance / (2 * sigma**2))
    return image


@pytest.fixture
def make_signal_image():
    """Return `generate_signal_image`, a factory for images of n-fold
    signals."""
    return generate_signal_image
