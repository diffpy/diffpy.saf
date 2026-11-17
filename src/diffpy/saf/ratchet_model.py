import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm

from diffpy.saf.filters import SymmetryAdaptedFilter


class RatchetModel:
    """Track the orientation of up to two signals through a stack of
    images.

    The signals may have any n-fold rotational symmetry. For each image, the
    orientation angle of the strongest signal (`phi1`) is found by
    maximizing its overlap with a symmetry adapted filter (SAF). If the
    curvature of the overlap shows a second signal, the first signal is
    masked out and the orientation angle of the second signal (`phi2`) is
    found the same way. Otherwise `phi2` follows `phi1`. The first image is
    searched over one full period of 360 / `n_folds` degrees. Every later
    image is searched only from `search_behind` degrees behind to
    `search_ahead` degrees ahead of the previous image's angles, so the
    angles accumulate as the sample rotates instead of wrapping back into
    one period.

    Parameters
    ----------
    n_folds : int
        The order of the rotational symmetry of the signals.
    resolution : float
        The angular resolution of the SAF, in degrees.
    search_ahead : float, optional
        The distance, in degrees, searched ahead of the previous angle in
        the direction of rotation given to `find_orientation_angles`.
        Defaults to half a period, 180 / `n_folds` degrees.
    search_behind : float, optional
        The distance, in degrees, searched behind the previous angle,
        against the direction of rotation given to
        `find_orientation_angles`. Defaults to half a period, 180 /
        `n_folds` degrees, so that with both defaults the search covers one
        full period centered on the previous angle.
    offset_step : float, optional
        The spacing, in degrees, of the angles searched. If a whole number of
        steps does not fit in one period of 360 / `n_folds` degrees, the
        step is adjusted to the nearest spacing that does. Default is 0.5.
    mask_width : float, optional
        The full angular width, in degrees, masked out around each lobe of
        the first signal when searching for the second. A width of 0
        masks nothing. Defaults to twice `resolution`.
    center : tuple of int, optional
        The position of the SAF center as (row, column). Defaults to the
        center of the images.

    Attributes
    ----------
    overlap1 : ndarray or None
        The overlap of each image with the SAF at `phi1` from the most
        recent call to `find_orientation_angles`, which is the maximum
        overlap found for the strongest signal. None until
        `find_orientation_angles` is called.
    overlap2 : ndarray or None
        The overlap of each image with the SAF at `phi2` from the most
        recent call to `find_orientation_angles`, taken after masking out
        the first signal whenever a second signal is found. It equals
        `overlap1` wherever no second signal is found. None until
        `find_orientation_angles` is called.
    """

    def __init__(
        self,
        n_folds,
        resolution,
        search_ahead=None,
        search_behind=None,
        offset_step=0.5,
        mask_width=None,
        center=None,
    ):
        self.n_folds = n_folds
        self.resolution = resolution
        half_period = 180 / n_folds
        if search_ahead is None:
            search_ahead = half_period
        if search_behind is None:
            search_behind = half_period
        self.search_ahead = search_ahead
        self.search_behind = search_behind
        self.offset_step = offset_step
        if mask_width is None:
            mask_width = 2 * resolution
        self.mask_width = mask_width
        self.center = center
        self.overlap1 = None
        self.overlap2 = None

    def find_orientation_angles(self, images, direction="ccw", verbose=True):
        """Find the orientation angles of up to two signals in each
        image.

        The overlap behind each angle is saved in the `overlap1` and
        `overlap2` attributes.

        Parameters
        ----------
        images : ndarray
            The images, either a single image of shape (rows, columns) or a
            stack of shape (n_images, rows, columns).
        direction : {"ccw", "cw"}, optional
            The direction of rotation. "ccw" means the angles increase from
            one image to the next, and "cw" means they decrease. It sets
            which way `search_ahead` and `search_behind` reach. Default is
            "ccw".
        verbose : bool, optional
            The switch for printing a progress bar over the images to
            stderr. Default is True.

        Returns
        -------
        phi1 : ndarray
            The orientation angle of the strongest signal in each image, in
            degrees.
        phi2 : ndarray
            The orientation angle of the second signal in each image, in
            degrees. It equals `phi1` wherever no second signal is found.

        Raises
        ------
        ValueError
            If `images` is neither a single 2D image nor a 3D stack, or if
            `direction` is not "ccw" or "cw".
        """
        if direction not in ("ccw", "cw"):
            raise ValueError(
                f"The direction must be 'ccw' or 'cw', but {direction!r} was "
                "given.\nPlease use 'ccw' if the angles increase from one "
                "image to the next, or 'cw' if they decrease."
            )
        images = self._get_image_stack(images)
        self._generate_search_filters(images.shape[1:])
        phi1, phi2 = [], []
        overlap1, overlap2 = [], []
        previous_phi1, previous_phi2 = 0.0, 0.0
        progress_bar = tqdm(
            images, desc="Finding orientation angles", disable=not verbose
        )
        for image_index, image in enumerate(progress_bar):
            is_first_image = image_index == 0
            overlap_curve1 = self._compute_overlap_optimized(image)
            window1 = self._get_search_window(
                previous_phi1, is_first_image, direction
            )
            best_phi1 = self._find_best_phi(overlap_curve1, window1)
            if self._has_second_signal(overlap_curve1):
                masked_image = self._mask_signal(image, best_phi1)
                overlap_curve2 = self._compute_overlap_optimized(masked_image)
                window2 = self._get_search_window(
                    previous_phi2, is_first_image, direction
                )
            else:
                overlap_curve2 = overlap_curve1
                window2 = self._get_period_window(previous_phi2)
            best_phi2 = self._find_best_phi(overlap_curve2, window2)
            phi1.append(best_phi1)
            phi2.append(best_phi2)
            overlap1.append(self._get_overlap_at(overlap_curve1, best_phi1))
            overlap2.append(self._get_overlap_at(overlap_curve2, best_phi2))
            previous_phi1, previous_phi2 = best_phi1, best_phi2
        self.overlap1 = np.array(overlap1)
        self.overlap2 = np.array(overlap2)
        phi1 = np.array(phi1)
        phi2 = np.array(phi2)
        return phi1, phi2

    def predict_direction(self, images, n_frames=10):
        """Predict the direction of rotation from the first images of a
        stack.

        The orientation angle of the strongest signal is found in each of
        the first `n_frames` images. Instead of `search_ahead` and
        `search_behind`, each image is searched over one full period
        centered on the previous image's angle, so the angles are followed
        in either direction and keep accumulating instead of wrapping back
        into one period. A line is fit to the angles, and the direction is
        `"ccw"` if its slope is positive and `"cw"` if it is negative. The
        signal must rotate by less than half a period, 180 / `n_folds`
        degrees, from one image to the next.

        Parameters
        ----------
        images : ndarray
            The stack of images, of shape (n_images, rows, columns).
        n_frames : int, optional
            The number of images, from the start of the stack, used for the
            prediction. Default is 10.

        Returns
        -------
        direction : {`"ccw"`, `"cw"`}
            The predicted direction of rotation.

        Raises
        ------
        ValueError
            If `images` is neither a single 2D image nor a 3D stack, if
            fewer than two images are used, or if the angle does not change
            over the images used.
        """
        images = self._get_image_stack(images)[:n_frames]
        if len(images) < 2:
            raise ValueError(
                "At least two images are needed to predict the direction of "
                f"rotation, but only {len(images)} was used.\nPlease pass a "
                "stack of two or more images and set n_frames to 2 or more."
            )
        self._generate_search_filters(images.shape[1:])
        phis = []
        previous_phi = 0.0
        for image in images:
            overlap_curve = self._compute_overlap_optimized(image)
            previous_phi = self._find_best_phi(
                overlap_curve, self._get_period_window(previous_phi)
            )
            phis.append(previous_phi)
        slope = np.polyfit(np.arange(len(phis)), phis, 1)[0]
        if np.isclose(slope, 0):
            raise ValueError(
                "The orientation angle did not change over the first "
                f"{len(images)} images, so the direction of rotation could "
                "not be predicted.\nPlease increase n_frames to include "
                "images where the sample rotates, or set the direction "
                "yourself."
            )
        direction = "ccw" if slope > 0 else "cw"
        return direction

    def compute_overlap(self, images, phis=None):
        """Compute the overlap of each image with the SAF as a function
        of orientation angle.

        The overlap at an angle is the sum over pixels of the image times
        the SAF with its first lobe along that angle, so it peaks at the
        orientation of each signal.

        Parameters
        ----------
        images : ndarray
            The images, either a single image of shape (rows, columns) or a
            stack of shape (n_images, rows, columns).
        phis : array-like, optional
            The orientation angles, in degrees, at which to compute the
            overlap. Defaults to one period, from 0 up to 360 / `n_folds`
            degrees in steps of `offset_step`.

        Returns
        -------
        phis : ndarray
            The orientation angles, in degrees.
        overlap : ndarray
            The overlap of each image at each angle, of shape
            (n_images, n_phis).

        Raises
        ------
        ValueError
            If `images` is neither a single 2D image nor a 3D stack.
        """
        images = self._get_image_stack(images)
        if phis is None:
            phis = self._get_period_phis(self.offset_step)
        phis = np.asarray(phis, dtype=float)
        filter_stack = self._generate_filter_stack(images.shape[1:], phis)
        overlap = np.tensordot(images, filter_stack, axes=([1, 2], [1, 2]))
        return phis, overlap

    def run_smoke_screen(self, image):
        """Find the orientation angles in one image and return the
        intermediate results.

        The image is searched the same way `find_orientation_angles`
        searches the first image of a stack, so the results show why `phi1`
        and `phi2` were found where they were. Signals are detected where
        the overlap curve bends downward, at local minima of its second
        derivative with negative values. Each angle found is shown with the
        mask `mask_width` degrees wide around each lobe of its SAF.

        Parameters
        ----------
        image : ndarray
            The image, of shape (rows, columns).

        Returns
        -------
        smoke_screen : dict
            The intermediate results, with the keys:

            - ``"phis"``
                The angles searched over one period, in degrees.
            - ``"overlap_curve"``
                The overlap of the image with the SAF at each angle in
                ``"phis"``.
            - ``"overlap_second_derivative"``
                The second derivative of ``"overlap_curve"`` with respect
                to the angle index, wrapping around the period.
            - ``"signal_phis"``
                The angles in ``"phis"`` where signals are detected.
            - ``"number_of_signals"``
                The number of signals detected.
            - ``"overlap_contrast"``
                The range of ``"overlap_curve"`` divided by its maximum,
                from 0 for an image with no preferred orientation to 1.
            - ``"phi1"``, ``"phi2"``
                The orientation angles of the strongest and second signal,
                in degrees.
            - ``"saf_opt1"``, ``"saf_opt2"``
                The SAF at ``"phi1"`` and ``"phi2"``.
            - ``"saf_opt1_mask"``, ``"saf_opt2_mask"``
                The pixels within `mask_width` / 2 degrees of a lobe of
                ``"saf_opt1"`` and ``"saf_opt2"``, as boolean arrays.
            - ``"saf_opt1_image"``, ``"saf_opt2_image"``
                The image times ``"saf_opt1_mask"`` and
                ``"saf_opt2_mask"``. For the second signal, the
                image has the first signal masked out whenever a second
                signal is detected.
            - ``"overlap1"``, ``"overlap2"``
                The overlap at ``"phi1"`` and ``"phi2"``, as saved by
                `find_orientation_angles`.
            - ``"signal_fraction1"``, ``"signal_fraction2"``
                The sums of ``"saf_opt1_image"`` and ``"saf_opt2_image"``
                divided by the sum of the image.
            - ``"masked_image"``
                The image with the first signal masked out.
            - ``"overlap_curve2"``
                The overlap curve searched for ``"phi2"``. It equals
                ``"overlap_curve"`` unless a second signal is detected.

        Raises
        ------
        ValueError
            If `image` is not a single 2D image.
        """
        image = np.asarray(image, dtype=float)
        if image.ndim != 2:
            raise ValueError(
                "The smoke screen runs on a single image of shape (rows, "
                f"columns), but an array of shape {image.shape} was given.\n"
                "Please pass one image, for example images[0] from a stack."
            )
        self._generate_search_filters(image.shape)
        overlap_curve = self._compute_overlap_optimized(image)
        second_derivative = self._compute_second_derivative(overlap_curve)
        signal_indices = self._find_signal_indices(overlap_curve)
        phi1 = self._find_best_phi(overlap_curve, self._get_period_window(0.0))
        masked_image = self._mask_signal(image, phi1)
        if len(signal_indices) > 1:
            image2 = masked_image
            overlap_curve2 = self._compute_overlap_optimized(masked_image)
            window2 = self._get_period_window(0.0)
        else:
            image2 = image
            overlap_curve2 = overlap_curve
            window2 = self._get_period_window(phi1)
        phi2 = self._find_best_phi(overlap_curve2, window2)
        smoke_screen = {
            "phis": self._phis,
            "overlap_curve": overlap_curve,
            "overlap_second_derivative": second_derivative,
            "signal_phis": self._phis[signal_indices],
            "number_of_signals": len(signal_indices),
            "overlap_contrast": np.ptp(overlap_curve) / np.max(overlap_curve),
            "masked_image": masked_image,
            "overlap_curve2": overlap_curve2,
        }
        for label, phi, search_image, search_curve in (
            ("1", phi1, image, overlap_curve),
            ("2", phi2, image2, overlap_curve2),
        ):
            saf_opt = self._filter_stack[self._get_period_index(phi)]
            saf_opt_mask = self._generate_signal_mask(image.shape, phi)
            saf_opt_image = np.where(saf_opt_mask, search_image, 0)
            smoke_screen[f"phi{label}"] = phi
            smoke_screen[f"saf_opt{label}"] = saf_opt
            smoke_screen[f"saf_opt{label}_mask"] = saf_opt_mask
            smoke_screen[f"saf_opt{label}_image"] = saf_opt_image
            smoke_screen[f"overlap{label}"] = self._get_overlap_at(
                search_curve, phi
            )
            smoke_screen[f"signal_fraction{label}"] = np.sum(
                saf_opt_image
            ) / np.sum(image)
        return smoke_screen

    @staticmethod
    def _get_image_stack(images):
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[np.newaxis]
        if images.ndim != 3:
            raise ValueError(
                "The images must be a single image of shape (rows, columns) "
                "or a stack of shape (n_images, rows, columns), but an array "
                f"of shape {images.shape} was given.\nPlease pass a 2D image "
                "or a 3D stack of images."
            )
        return images

    def _get_period_phis(self, step):
        """Return evenly spaced angles over one period, with the spacing
        adjusted to the nearest one that fits a whole number of
        times."""
        period = 360 / self.n_folds
        n_steps = max(int(round(period / step)), 1)
        period_phis = np.arange(n_steps) * period / n_steps
        return period_phis

    def _generate_filter_stack(self, imshape, phis):
        """Return the SAF at each angle in `phis`, stacked along axis
        0."""
        saf = SymmetryAdaptedFilter(
            self.resolution, self.n_folds, imshape=imshape, center=self.center
        )
        filters = []
        for phi in phis:
            saf.phi_0 = phi
            filters.append(saf.generate_filter())
        filter_stack = np.array(filters)
        return filter_stack

    def _generate_search_filters(self, imshape):
        """Build the SAFs searched by `find_orientation_angles`, once
        per call, at every angle in one period spaced by
        `offset_step`."""
        self._phis = self._get_period_phis(self.offset_step)
        self._phi_step = 360 / self.n_folds / len(self._phis)
        self._filter_stack = self._generate_filter_stack(imshape, self._phis)

    def _compute_overlap_optimized(self, image):
        """Return the overlap of `image` at each angle in one period,
        reusing the SAFs built by `_generate_search_filters` instead of
        building them again like `compute_overlap` does."""
        overlap_curve = np.tensordot(
            self._filter_stack, image, axes=([1, 2], [0, 1])
        )
        return overlap_curve

    def _get_overlap_at(self, overlap_curve, phis):
        """Look up the overlap at any angles, using that the overlap
        curve repeats every period."""
        overlaps = overlap_curve[self._get_period_index(phis)]
        return overlaps

    def _get_period_index(self, phis):
        """Return the position of each angle within one period of
        `self._phis`."""
        steps_from_zero = np.round(np.asarray(phis) / self._phi_step)
        period_index = steps_from_zero.astype(int) % len(self._phis)
        return period_index

    def _find_best_phi(self, overlap_curve, window):
        """Return the angle in `window` with the largest overlap."""
        overlaps = self._get_overlap_at(overlap_curve, window)
        best_phi = window[np.argmax(overlaps)]
        return best_phi

    def _get_search_window(self, previous_phi, is_first_image, direction):
        """Return the angles to search, in degrees.

        The first image is searched over a full period around 0 degrees,
        and every later image within `search_behind` and `search_ahead`
        around `previous_phi`.
        """
        if is_first_image:
            search_window = self._get_period_window(0.0)
        else:
            search_window = self._get_tracking_window(previous_phi, direction)
        return search_window

    def _get_period_window(self, center_phi):
        """Return one period of angles centered on `center_phi`."""
        period = 360 / self.n_folds
        period_window = self._get_grid_window(
            center_phi - period / 2, center_phi + period / 2
        )
        return period_window

    def _get_tracking_window(self, previous_phi, direction):
        """Return the angles from `search_behind` degrees behind to
        `search_ahead` degrees ahead of `previous_phi`, in the direction
        of rotation."""
        if direction == "ccw":
            low = previous_phi - self.search_behind
            high = previous_phi + self.search_ahead
        else:
            low = previous_phi - self.search_ahead
            high = previous_phi + self.search_behind
        tracking_window = self._get_grid_window(low, high)
        return tracking_window

    def _get_grid_window(self, low, high):
        """Return the angles from `low` up to, but not including, `high`
        that lie on the grid of the SAF stack, so every angle has a
        precomputed SAF."""
        first_step = np.ceil(round(low / self._phi_step, 9))
        last_step = np.ceil(round(high / self._phi_step, 9))
        grid_window = np.arange(first_step, last_step) * self._phi_step
        return grid_window

    @staticmethod
    def _has_second_signal(overlap_curve):
        has_second_signal = (
            RatchetModel.count_number_of_signals(overlap_curve) > 1
        )
        return has_second_signal

    @staticmethod
    def count_number_of_signals(overlap_curve):
        """Count the signals in one period of the overlap curve.

        Each signal makes the overlap bend downward, so it shows up as a
        local maximum of positive curvature, the negative second
        derivative. Local maxima with negative curvature sit in the
        valleys between signals and are not counted. The curve repeats
        every period, so the second derivative and the maximum search
        both wrap around its ends.
        """
        number_of_signals = len(
            RatchetModel._find_signal_indices(overlap_curve)
        )
        return number_of_signals

    @staticmethod
    def _find_signal_indices(overlap_curve):
        """Return the indices of the local maxima of positive curvature
        in one period of the overlap curve."""
        curvature = -RatchetModel._compute_second_derivative(overlap_curve)
        maxima = argrelextrema(curvature, np.greater, mode="wrap")[0]
        signal_indices = maxima[curvature[maxima] > 0]
        return signal_indices

    @staticmethod
    def _compute_second_derivative(overlap_curve):
        """Return the second derivative of one period of the overlap
        curve, wrapping around its ends."""
        second_derivative = (
            np.roll(overlap_curve, -1)
            - 2 * overlap_curve
            + np.roll(overlap_curve, 1)
        )
        return second_derivative

    def mask_with_annulus(self, images, inner_radius, outer_radius):
        """Zero the pixels of each image outside an annulus around
        `center`.

        Pixels whose distance from `center` lies between `inner_radius`
        and `outer_radius`, inclusive, keep their values and all other
        pixels are set to 0.

        Parameters
        ----------
        images : ndarray
            The images, either a single image of shape (rows, columns) or a
            stack of shape (n_images, rows, columns).
        inner_radius : float
            The inner radius of the annulus, in pixels.
        outer_radius : float
            The outer radius of the annulus, in pixels.

        Returns
        -------
        masked_images : ndarray
            The masked images, of shape (n_images, rows, columns).

        Raises
        ------
        ValueError
            If `images` is neither a single 2D image nor a 3D stack.
        """
        images = self._get_image_stack(images)
        n_rows, n_cols = images.shape[1:]
        y_grid, x_grid = np.mgrid[0:n_rows, 0:n_cols]
        center = self.center
        if center is None:
            center = (n_rows // 2, n_cols // 2)
        center_row, center_col = center
        distance_from_center = np.sqrt(
            (y_grid - center_row) ** 2 + (x_grid - center_col) ** 2
        )
        annular_mask = (distance_from_center >= inner_radius) & (
            distance_from_center <= outer_radius
        )
        masked_images = np.where(annular_mask, images, 0)
        return masked_images

    def _generate_signal_mask(self, imshape, phi):
        """Return the mask `mask_width` degrees wide around each lobe of
        the SAF at `phi`."""
        saf = SymmetryAdaptedFilter(
            self.resolution,
            self.n_folds,
            imshape=imshape,
            phi_0=phi,
            center=self.center,
        )
        signal_mask = saf.generate_mask(self.mask_width)
        return signal_mask

    def _mask_signal(self, image, phi):
        """Zero the pixels of `image` within `mask_width` / 2 degrees of
        a lobe of the SAF at `phi`."""
        signal_mask = self._generate_signal_mask(image.shape, phi)
        masked_image = np.where(signal_mask, 0, image)
        return masked_image
