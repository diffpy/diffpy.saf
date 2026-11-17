import re

import numpy as np
import pytest

from diffpy.saf.filters import SymmetryAdaptedFilter

# The filter has n_folds lobes, at phi_0 + m * 360 / n_folds degrees from
# the center, where angles run from the +column axis toward the +row axis.
# Each case reads an 11x11 filter at one pixel on its edge or corner: 1
# means the pixel sits on a lobe, 0 means it sits halfway between two lobes.
# With the default center (5, 5), the edge and corner pixels lie along these
# directions:
#
#              column 0   column 5   column 10
#     row 0      225        270        315
#     row 5      180       center       0
#     row 10     135         90        45


@pytest.mark.parametrize(
    "input_n_folds, input_phi_0, input_imshape, input_center, input_pixel, "
    "expected_value",
    [
        # C1: 6-fold filter with no rotation, pixel along 0 degrees.
        # Expected: The pixel sits on a lobe.
        (6, 0.0, (11, 11), None, (5, 10), 1.0),
        # C2: 6-fold filter with no rotation, pixel along 90 degrees
        # (halfway between the lobes at 60 and 120 degrees).
        # Expected: The pixel sits between lobes.
        (6, 0.0, (11, 11), None, (10, 5), 0.0),
        # C3: 2-fold filter rotated by 45 degrees, pixel along 45 degrees.
        # Expected: The pixel sits on a lobe, so phi_0 is read in degrees
        # and rotates the lobes toward the +row axis.
        (2, 45.0, (11, 11), None, (10, 10), 1.0),
        # C4: 2-fold filter rotated by 45 degrees, pixel along 135 degrees
        # (the mirror image of C3).
        # Expected: The pixel sits between lobes, so the lobes are not
        # rotated the opposite way.
        (2, 45.0, (11, 11), None, (10, 0), 0.0),
        # C5: 4-fold filter rotated by a full period (90 degrees), pixel
        # along 0 degrees.
        # Expected: The pixel sits on a lobe, since the lobes repeat every
        # period.
        (4, 90.0, (11, 11), None, (5, 10), 1.0),
        # C6: Non-square 11x21 filter with the default center (5, 10), pixel
        # along 0 degrees.
        # Expected: The pixel sits on a lobe, so the default center is
        # (imshape[0] // 2, imshape[1] // 2) as (row, column).
        (6, 0.0, (11, 21), None, (5, 20), 1.0),
        # C7: Explicit center at the top middle pixel (0, 5), 2-fold filter
        # rotated by 45 degrees, pixel along 45 degrees from that center.
        # Expected: The pixel sits on a lobe.
        (2, 45.0, (11, 11), (0, 5), (5, 10), 1.0),
        # C8: Explicit center at the top middle pixel (0, 5), 2-fold filter
        # rotated by 45 degrees, pixel along 135 degrees from that center.
        # Expected: The pixel sits between lobes.
        (2, 45.0, (11, 11), (0, 5), (5, 0), 0.0),
    ],
)
def test_symmetry_adapted_filter_lobe_directions(
    input_n_folds,
    input_phi_0,
    input_imshape,
    input_center,
    input_pixel,
    expected_value,
):
    saf = SymmetryAdaptedFilter(
        resolution=10,
        n_folds=input_n_folds,
        imshape=input_imshape,
        phi_0=input_phi_0,
        center=input_center,
    )
    actual_filter = saf.generate_filter()
    actual_value = actual_filter[input_pixel]
    assert actual_filter.shape == input_imshape
    assert actual_value == pytest.approx(expected_value, abs=1e-9)


# The mask is 1 at every pixel whose direction from the center lies within
# width / 2 degrees of a lobe center, and 0 everywhere else. The expected
# mask is built from each pixel's angular distance to the nearest lobe,
# where the lobes sit every 360 / n_folds degrees from phi_0.


@pytest.mark.parametrize(
    "input_n_folds, input_phi_0, input_width",
    [
        # C1: 6-fold mask with no rotation, 20 degrees wide.
        # Expected: 1 within 10 degrees of each lobe, 0 elsewhere.
        (6, 0.0, 20),
        # C2: 2-fold mask rotated by 40 degrees, 8 degrees wide.
        # Expected: 1 within 4 degrees of each lobe, 0 elsewhere.
        (2, 40.0, 8),
        # C3: 4-fold mask rotated by 10 degrees, 60 degrees wide.
        # Expected: 1 within 30 degrees of each lobe, 0 elsewhere.
        (4, 10.0, 60),
        # C4: 6-fold mask 0 degrees wide.
        # Expected: 0 everywhere, since no pixel is within 0 degrees.
        (6, 0.0, 0),
    ],
)
def test_generate_mask(input_n_folds, input_phi_0, input_width):
    saf = SymmetryAdaptedFilter(
        resolution=10,
        n_folds=input_n_folds,
        imshape=(51, 51),
        phi_0=input_phi_0,
    )
    row_grid, col_grid = np.mgrid[0:51, 0:51]
    pixel_phis = np.rad2deg(np.arctan2(row_grid - 25, col_grid - 25))
    period = 360 / input_n_folds
    distance_to_lobe = np.abs(
        (pixel_phis - input_phi_0 + period / 2) % period - period / 2
    )
    expected_mask = distance_to_lobe < input_width / 2
    actual_mask = saf.generate_mask(input_width)
    assert np.array_equal(actual_mask, expected_mask)


@pytest.mark.parametrize(
    "input_width, expected_msg",
    [
        # C1: A negative width of -5 degrees for a 6-fold filter.
        # Expected: A ValueError gives the allowed range of widths.
        (
            -5,
            "The mask width must be from 0 to one period of 360 / n_folds "
            "= 60 degrees, but -5 was given.\n"
            "Please choose a width from 0, for no masking, to 60 degrees.",
        ),
        # C2: A width of 70 degrees for a 6-fold filter, more than one
        # period.
        # Expected: A ValueError gives the allowed range of widths.
        (
            70,
            "The mask width must be from 0 to one period of 360 / n_folds "
            "= 60 degrees, but 70 was given.\n"
            "Please choose a width from 0, for no masking, to 60 degrees.",
        ),
    ],
)
def test_generate_mask_bad_width(input_width, expected_msg):
    saf = SymmetryAdaptedFilter(resolution=10, n_folds=6, imshape=(11, 11))
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        saf.generate_mask(input_width)
