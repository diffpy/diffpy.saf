import re

import numpy as np
import pytest

from diffpy.saf.filters import SymmetryAdaptedFilter
from diffpy.saf.ratchet_model import RatchetModel

# The ratchet model follows up to two n-fold signals through a stack of images.
# phi1 tracks the strongest signal. phi2 tracks the second signal when one is
# present, and otherwise follows phi1. After the first frame, each angle is
# searched only from search_behind degrees behind to search_ahead degrees ahead
# of its value in the previous frame, so the angles keep accumulating instead
# of wrapping back into one period of 360 / n_folds degrees. Every case rotates
# the signals by 8 degrees per frame, which is inside the 10 degree
# search_ahead but outside the 5 degree search_behind, so the signals can only
# be followed if the window opens in the given direction of rotation.


@pytest.mark.parametrize(
    "input_n_folds, input_start_angles, input_amplitudes, input_step, "
    "input_n_frames, input_direction, expected_phi1, expected_phi2",
    [
        # C1: One 6-fold signal rotating toward increasing angles, tracked
        # in the "ccw" direction.
        # Expected: phi1 follows the signal, and phi2 matches phi1.
        (6, [0], [1.0], 8, 6, "ccw", 8 * np.arange(6), 8 * np.arange(6)),
        # C2: One 6-fold signal rotating toward decreasing angles, tracked
        # in the "cw" direction.
        # Expected: phi1 follows the signal, and phi2 matches phi1.
        (6, [0], [1.0], -8, 6, "cw", -8 * np.arange(6), -8 * np.arange(6)),
        # C3: One 6-fold signal rotating from 0 to 80 degrees, past its 60
        # degree period.
        # Expected: The angles keep accumulating past 60 degrees instead of
        # wrapping back to 0.
        (6, [0], [1.0], 8, 11, "ccw", 8 * np.arange(11), 8 * np.arange(11)),
        # C4: One 4-fold signal rotating from 0 to 96 degrees, past its 90
        # degree period.
        # Expected: The angles keep accumulating past 90 degrees instead of
        # wrapping back to 0.
        (4, [0], [1.0], 8, 13, "ccw", 8 * np.arange(13), 8 * np.arange(13)),
        # C5: One 3-fold signal rotating toward increasing angles.
        # Expected: phi1 follows the signal, and phi2 matches phi1.
        (3, [0], [1.0], 8, 6, "ccw", 8 * np.arange(6), 8 * np.arange(6)),
        # C6: One 2-fold signal rotating toward increasing angles.
        # Expected: phi1 follows the signal, and phi2 matches phi1.
        (2, [0], [1.0], 8, 6, "ccw", 8 * np.arange(6), 8 * np.arange(6)),
        # C7: Two 6-fold signals 25 degrees apart rotating together, with
        # the signal at 0 degrees the stronger of the two.
        # Expected: phi1 follows the stronger signal and phi2 follows the
        # weaker signal.
        (
            6,
            [0, 25],
            [1.0, 0.6],
            8,
            6,
            "ccw",
            8 * np.arange(6),
            25 + 8 * np.arange(6),
        ),
        # C8: Two 4-fold signals 40 degrees apart rotating together, with
        # the signal at 0 degrees the stronger of the two.
        # Expected: phi1 follows the stronger signal and phi2 follows the
        # weaker signal.
        (
            4,
            [0, 40],
            [1.0, 0.6],
            8,
            6,
            "ccw",
            8 * np.arange(6),
            40 + 8 * np.arange(6),
        ),
    ],
)
def test_find_orientation_angles(
    make_signal_image,
    input_n_folds,
    input_start_angles,
    input_amplitudes,
    input_step,
    input_n_frames,
    input_direction,
    expected_phi1,
    expected_phi2,
):
    input_images = np.array(
        [
            make_signal_image(
                input_n_folds,
                np.add(input_start_angles, frame_index * input_step),
                input_amplitudes,
            )
            for frame_index in range(input_n_frames)
        ]
    )
    model = RatchetModel(
        n_folds=input_n_folds,
        resolution=10,
        search_ahead=10,
        search_behind=5,
    )
    actual_phi1, actual_phi2 = model.find_orientation_angles(
        input_images, direction=input_direction
    )
    assert actual_phi1 == pytest.approx(expected_phi1, abs=1e-6)
    assert actual_phi2 == pytest.approx(expected_phi2, abs=1e-6)


# Each of search_ahead and search_behind not given defaults to half a period,
# 180 / n_folds degrees, so together they cover one full period centered on the
# previous angle. The signals can then be followed in either direction,
# whatever direction is passed to find_orientation_angles, as long as they
# rotate by less than half a period per frame.


@pytest.mark.parametrize(
    "input_n_folds, input_search_distances, expected_search_ahead, "
    "expected_search_behind",
    [
        # C1: A 6-fold model with neither search distance given.
        # Expected: Both search distances are 30 degrees, half the 60 degree
        # period.
        (6, {}, 30, 30),
        # C2: A 4-fold model with neither search distance given.
        # Expected: Both search distances are 45 degrees, half the 90 degree
        # period.
        (4, {}, 45, 45),
        # C3: A 6-fold model with only search_ahead given.
        # Expected: search_ahead is kept, and search_behind is 30 degrees.
        (6, {"search_ahead": 10}, 10, 30),
        # C4: A 6-fold model with only search_behind given.
        # Expected: search_behind is kept, and search_ahead is 30 degrees.
        (6, {"search_behind": 5}, 30, 5),
        # C5: A 6-fold model with both search distances given.
        # Expected: Both search distances are kept.
        (6, {"search_ahead": 10, "search_behind": 5}, 10, 5),
    ],
)
def test_ratchet_model_default_search_distances(
    input_n_folds,
    input_search_distances,
    expected_search_ahead,
    expected_search_behind,
):
    model = RatchetModel(
        n_folds=input_n_folds, resolution=10, **input_search_distances
    )
    actual_search_ahead = model.search_ahead
    actual_search_behind = model.search_behind
    assert actual_search_ahead == pytest.approx(expected_search_ahead)
    assert actual_search_behind == pytest.approx(expected_search_behind)


@pytest.mark.parametrize(
    "input_n_folds, input_step, input_n_frames, expected_phi1",
    [
        # C1: One 6-fold signal rotating toward increasing angles by 8
        # degrees per frame, past its 60 degree period.
        # Expected: phi1 follows the signal and keeps accumulating past 60
        # degrees.
        (6, 8, 11, 8 * np.arange(11)),
        # C2: One 6-fold signal rotating toward decreasing angles by 8
        # degrees per frame, while the direction is left at its default
        # "ccw".
        # Expected: phi1 still follows the signal, since the default window
        # reaches equally far in both directions.
        (6, -8, 11, -8 * np.arange(11)),
        # C3: One 4-fold signal rotating by 20 degrees per frame, within
        # half its 90 degree period.
        # Expected: phi1 follows the signal.
        (4, 20, 6, 20 * np.arange(6)),
    ],
)
def test_find_orientation_angles_default_search_distances(
    make_signal_image, input_n_folds, input_step, input_n_frames, expected_phi1
):
    input_images = np.array(
        [
            make_signal_image(input_n_folds, [frame_index * input_step], [1.0])
            for frame_index in range(input_n_frames)
        ]
    )
    model = RatchetModel(n_folds=input_n_folds, resolution=10)
    actual_phi1, _ = model.find_orientation_angles(input_images)
    assert actual_phi1 == pytest.approx(expected_phi1, abs=1e-6)


# A single 2D image is treated as a stack of one image, so the angles are
# returned as arrays of length one.


@pytest.mark.parametrize(
    "input_n_folds, input_start_angles, input_amplitudes, expected_phi1, "
    "expected_phi2",
    [
        # C1: A single image of one 6-fold signal at 10 degrees.
        # Expected: phi1 and phi2 are both [10].
        (6, [10], [1.0], [10], [10]),
        # C2: A single image of two 4-fold signals at 0 and 40 degrees,
        # with the signal at 0 degrees the stronger of the two.
        # Expected: phi1 is [0] and phi2 is [40].
        (4, [0, 40], [1.0, 0.6], [0], [40]),
    ],
)
def test_find_orientation_angles_single_image(
    make_signal_image,
    input_n_folds,
    input_start_angles,
    input_amplitudes,
    expected_phi1,
    expected_phi2,
):
    input_image = make_signal_image(
        input_n_folds, input_start_angles, input_amplitudes
    )
    model = RatchetModel(
        n_folds=input_n_folds,
        resolution=10,
        search_ahead=10,
        search_behind=5,
    )
    actual_phi1, actual_phi2 = model.find_orientation_angles(input_image)
    assert actual_phi1 == pytest.approx(expected_phi1, abs=1e-6)
    assert actual_phi2 == pytest.approx(expected_phi2, abs=1e-6)


# The direction of rotation is predicted from the first n_frames images of a
# stack. Each image is searched over one full period centered on the previous
# angle, so the angles are followed in either direction without search_ahead
# and search_behind and keep accumulating past one period of 360 / n_folds
# degrees. A line is fit to the angles of the strongest signal, and a positive
# slope means "ccw" while a negative slope means "cw".


@pytest.mark.parametrize(
    "input_n_folds, input_angles, input_n_frames, expected_direction",
    [
        # C1: One 6-fold signal rotating toward increasing angles by 8
        # degrees per frame.
        # Expected: The direction is "ccw".
        (6, 8 * np.arange(10), 10, "ccw"),
        # C2: One 6-fold signal rotating toward decreasing angles by 8
        # degrees per frame.
        # Expected: The direction is "cw".
        (6, -8 * np.arange(10), 10, "cw"),
        # C3: One 4-fold signal rotating toward decreasing angles by 15
        # degrees per frame.
        # Expected: The direction is "cw".
        (4, -15 * np.arange(10), 10, "cw"),
        # C4: One 6-fold signal rotating by 20 degrees per frame, beyond any
        # usual search distance but within half its 60 degree period.
        # Expected: The direction is "ccw".
        (6, 20 * np.arange(10), 10, "ccw"),
        # C5: One 6-fold signal rotating from 0 to 152 degrees, well past
        # its 60 degree period, with only the default first 10 frames used.
        # Expected: The angles keep accumulating instead of wrapping back
        # to 0, and the direction is "ccw".
        (6, 8 * np.arange(20), None, "ccw"),
        # C6: One 6-fold signal that rotates toward increasing angles for 5
        # frames and then toward decreasing angles, with only the first 5
        # frames used.
        # Expected: The later frames are ignored, and the direction is
        # "ccw".
        (6, [0, 8, 16, 24, 32, 24, 16, 8, 0, -8], 5, "ccw"),
        # C7: A stack of only 2 frames, shorter than the default 10.
        # Expected: Both frames are used, and the direction is "cw".
        (6, [0, -8], None, "cw"),
    ],
)
def test_predict_direction(
    make_signal_image,
    input_n_folds,
    input_angles,
    input_n_frames,
    expected_direction,
):
    input_images = np.array(
        [
            make_signal_image(input_n_folds, [angle], [1.0])
            for angle in input_angles
        ]
    )
    model = RatchetModel(
        n_folds=input_n_folds,
        resolution=10,
        search_ahead=10,
        search_behind=5,
    )
    if input_n_frames is None:
        actual_direction = model.predict_direction(input_images)
    else:
        actual_direction = model.predict_direction(
            input_images, n_frames=input_n_frames
        )
    assert actual_direction == expected_direction


@pytest.mark.parametrize(
    "input_angles, input_n_frames, expected_msg",
    [
        # C1: A single image, so there is no change in angle to fit.
        # Expected: A ValueError explains that at least two images are
        # needed.
        (
            [0],
            10,
            "At least two images are needed to predict the direction of "
            "rotation, but only 1 was used.\n"
            "Please pass a stack of two or more images and set n_frames to "
            "2 or more.",
        ),
        # C2: A stack of images with n_frames set to 1.
        # Expected: A ValueError explains that at least two images are
        # needed.
        (
            [0, 8, 16],
            1,
            "At least two images are needed to predict the direction of "
            "rotation, but only 1 was used.\n"
            "Please pass a stack of two or more images and set n_frames to "
            "2 or more.",
        ),
        # C3: A signal that does not rotate.
        # Expected: A ValueError explains that the angle did not change.
        (
            [10, 10, 10, 10],
            10,
            "The orientation angle did not change over the first 4 images, "
            "so the direction of rotation could not be predicted.\n"
            "Please increase n_frames to include images where the sample "
            "rotates, or set the direction yourself.",
        ),
    ],
)
def test_predict_direction_bad_input(
    make_signal_image, input_angles, input_n_frames, expected_msg
):
    input_images = np.array(
        [make_signal_image(6, [angle], [1.0]) for angle in input_angles]
    )
    model = RatchetModel(
        n_folds=6, resolution=10, search_ahead=10, search_behind=5
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model.predict_direction(input_images, n_frames=input_n_frames)


# The overlap at an orientation angle is the sum over pixels of the image
# times the SAF with its first lobe along that angle, so for each image it
# peaks at the orientation of the signal. By default it is computed over one
# period, from 0 up to 360 / n_folds degrees in steps of offset_step, but any
# angles can be passed instead, including angles outside one period.


@pytest.mark.parametrize(
    "input_n_folds, input_start_angles, input_step, input_n_frames, "
    "input_phis, expected_phis, expected_peak_phis",
    [
        # C1: One image of a 6-fold signal at 10 degrees, default angles.
        # Expected: The overlap covers 0 to 60 degrees in 0.5 degree steps
        # and peaks at 10 degrees.
        (6, [10], 0, 1, None, np.arange(0, 60, 0.5), [10]),
        # C2: The same image with angles from -20 to 40 degrees in 1 degree
        # steps.
        # Expected: The overlap is computed at the given angles and peaks at
        # 10 degrees.
        (6, [10], 0, 1, np.arange(-20, 40, 1), np.arange(-20, 40, 1), [10]),
        # C3: The same image with angles from 40 to 100 degrees, outside
        # the first period.
        # Expected: The overlap peaks at 70 degrees, one period after the
        # signal at 10 degrees.
        (6, [10], 0, 1, np.arange(40, 100, 1), np.arange(40, 100, 1), [70]),
        # C4: A stack of three images of a 4-fold signal rotating by 8
        # degrees per image, default angles.
        # Expected: One overlap curve per image, covering 0 to 90 degrees in
        # 0.5 degree steps, each peaking at that image's orientation.
        (4, [0], 8, 3, None, np.arange(0, 90, 0.5), [0, 8, 16]),
    ],
)
def test_compute_overlap(
    make_signal_image,
    input_n_folds,
    input_start_angles,
    input_step,
    input_n_frames,
    input_phis,
    expected_phis,
    expected_peak_phis,
):
    input_images = np.array(
        [
            make_signal_image(
                input_n_folds,
                np.add(input_start_angles, frame_index * input_step),
                [1.0],
            )
            for frame_index in range(input_n_frames)
        ]
    )
    model = RatchetModel(
        n_folds=input_n_folds,
        resolution=10,
        search_ahead=10,
        search_behind=5,
    )
    actual_phis, actual_overlap = model.compute_overlap(
        input_images, phis=input_phis
    )
    actual_peak_phis = actual_phis[np.argmax(actual_overlap, axis=1)]
    assert actual_phis == pytest.approx(expected_phis)
    assert actual_peak_phis == pytest.approx(expected_peak_phis)


def get_saf_overlap(image, n_folds, phi, masked_phi=None):
    """Return the overlap of an image with the SAF at `phi`, after
    masking out the signal at `masked_phi` if one is given."""
    if masked_phi is not None:
        signal_mask = SymmetryAdaptedFilter(
            10, n_folds, image.shape, phi_0=masked_phi
        ).generate_mask(20)
        image = np.where(signal_mask, 0, image)
    saf = SymmetryAdaptedFilter(10, n_folds, image.shape, phi_0=phi)
    overlap = np.sum(image * saf.generate_filter())
    return overlap


# The number of signals is counted from one period of the overlap curve of
# an image with two 6-fold signals: a strong signal at 0 degrees and a
# weaker signal rotated away from it. Every angle difference below is
# combined with every weak signal amplitude. The angle differences get
# finer as the signals get closer, down to well below the 10 degree
# resolution of the SAF, and the weak signal should always be counted.
# The angles phi1 and phi2 should also land on the two signals. They are
# compared in sorted order, since with equal amplitudes either signal may
# be found first. A single image is searched over [-30, 30) degrees,
# so a signal at 30 degrees is reported at -30 degrees, the same
# orientation one period earlier.


@pytest.mark.parametrize(
    "input_signal2_phi",
    [
        # Far apart, in 5 degree steps.
        # C1: The weak signal 30 degrees from the strong signal, halfway
        # between two of its lobes.
        30,
        # C2: The weak signal 25 degrees from the strong signal.
        25,
        # C3: The weak signal 20 degrees from the strong signal.
        20,
        # C4: The weak signal 15 degrees from the strong signal.
        15,
        # Around the 10 degree resolution of the SAF, in 2 degree steps.
        # C5: The weak signal 12 degrees from the strong signal.
        12,
    ],
)
@pytest.mark.parametrize(
    "input_signal2_amplitude, expected_number_of_signals",
    [
        # C1: The weak signal is a quarter as bright as the strong signal.
        # Expected: Two signals are counted.
        (0.25, 2),
        # C2: The weak signal is half as bright as the strong signal.
        # Expected: Two signals are counted.
        (0.5, 2),
        # C3: The weak signal is three quarters as bright as the strong
        # signal.
        # Expected: Two signals are counted.
        (0.75, 2),
        # C4: Both signals are equally bright.
        # Expected: Two signals are counted.
        (1.0, 2),
    ],
)
def test_count_number_of_signals(
    make_signal_image,
    input_signal2_phi,
    input_signal2_amplitude,
    expected_number_of_signals,
):
    # Generating the image from the fixture
    sigma_signals = 1.5
    n_folds_signals = 6
    signal1_phi = 0
    signal1_amplitude = 1
    # generate a single image with two 6-fold signals,
    # one at 0 degrees and the other rotated away by
    # input_signal2_phi, with the second signal weaker
    # by input_signal2_amplitude
    input_image = make_signal_image(
        n_folds_signals,
        [signal1_phi, input_signal2_phi],
        [signal1_amplitude, input_signal2_amplitude],
        sigma=sigma_signals,
    )
    # model setup, masking 15 degrees around the strong signal since its
    # peaks are wider in angle than the 4 degree resolution
    model = RatchetModel(
        n_folds=6,
        resolution=4,
        search_ahead=10,
        search_behind=5,
        mask_width=15,
    )
    model._generate_search_filters(input_image.shape)
    overlap_curve = model._compute_overlap_optimized(input_image)

    # extract number of signals
    actual_number_of_signals = model.count_number_of_signals(overlap_curve)
    actual_phi1, actual_phi2 = model.find_orientation_angles(input_image)
    actual_phis = sorted([actual_phi1[0], actual_phi2[0]])
    # modulo to get the equivalent angle in the range [-30, 30)
    input_phi2_mod = (input_signal2_phi + 30) % 60 - 30
    expected_phis = sorted([0, input_phi2_mod])
    assert actual_number_of_signals == expected_number_of_signals
    assert actual_phis == pytest.approx(expected_phis, abs=3)  # ±3 deg


# With only one signal present, nothing else in the overlap curve should be
# counted as a second signal.


@pytest.mark.parametrize(
    "input_start_angle, expected_number_of_signals",
    [
        # C1: A single 6-fold signal at 0 degrees.
        # Expected: One signal is counted.
        (0, 1),
    ],
)
def test_count_number_of_signals_with_one_signal(
    make_signal_image, input_start_angle, expected_number_of_signals
):
    input_image = make_signal_image(6, [input_start_angle], [1.0])
    model = RatchetModel(
        n_folds=6, resolution=10, search_ahead=10, search_behind=5
    )
    model._generate_search_filters(input_image.shape)
    overlap_curve = model._compute_overlap_optimized(input_image)
    actual_number_of_signals = model.count_number_of_signals(overlap_curve)
    assert actual_number_of_signals == expected_number_of_signals


def test_find_orientation_angles_bad_direction(make_signal_image):
    # C1: A direction other than "ccw" or "cw" is given.
    # Expected: A ValueError explains the two allowed values.
    input_image = make_signal_image(6, [0], [1.0])
    model = RatchetModel(
        n_folds=6, resolution=10, search_ahead=10, search_behind=5
    )
    expected_msg = (
        "The direction must be 'ccw' or 'cw', but 'up' was given.\n"
        "Please use 'ccw' if the angles increase from one image to the "
        "next, or 'cw' if they decrease."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model.find_orientation_angles(input_image, direction="up")


@pytest.mark.parametrize(
    "input_shape, expected_msg",
    [
        # C1: A 1D array.
        # Expected: A ValueError explains that a 2D image or a 3D stack is
        # needed.
        (
            (64,),
            "The images must be a single image of shape (rows, columns) or "
            "a stack of shape (n_images, rows, columns), but an array of "
            "shape (64,) was given.\n"
            "Please pass a 2D image or a 3D stack of images.",
        ),
        # C2: A 4D array.
        # Expected: A ValueError explains that a 2D image or a 3D stack is
        # needed.
        (
            (2, 3, 64, 64),
            "The images must be a single image of shape (rows, columns) or "
            "a stack of shape (n_images, rows, columns), but an array of "
            "shape (2, 3, 64, 64) was given.\n"
            "Please pass a 2D image or a 3D stack of images.",
        ),
    ],
)
def test_find_orientation_angles_bad_input_dimension(
    input_shape, expected_msg
):
    model = RatchetModel(
        n_folds=6, resolution=10, search_ahead=10, search_behind=5
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model.find_orientation_angles(np.zeros(input_shape))
