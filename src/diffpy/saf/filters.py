import numpy as np


class SymmetryAdaptedFilter:
    """Build an angular filter with n-fold rotational symmetry.

    The filter has `n_folds` lobes radiating from `center`, centered on
    the directions `phi_0 + m * 360 / n_folds` degrees. Angles are measured
    from the +column axis toward the +row axis, which appears clockwise
    when the image is shown with its origin in the upper left.

    Parameters
    ----------
    resolution : float
        The angular resolution, in degrees, that sets the width of each
        lobe.
    n_folds : int
        The number of lobes, i.e. the order of the rotational symmetry.
    imshape : tuple of int, optional
        The shape of the filter as (rows, columns). Default is (128, 128).
    phi_0 : float, optional
        The rotation angle of the filter in degrees. Default is 0.
    center : tuple of int, optional
        The position of the filter center as (row, column). Defaults to
        `(imshape[0] // 2, imshape[1] // 2)`.
    """

    def __init__(
        self,
        resolution,
        n_folds,
        imshape=(128, 128),
        phi_0=0.0,
        center=None,
    ):
        self.k = self._find_k_value(resolution, n_folds)
        self.n_folds = n_folds
        self.phi_0 = phi_0
        self.imshape = imshape
        if center is None:
            center = (imshape[0] // 2, imshape[1] // 2)
        self.center = center

    @staticmethod
    def _find_k_value(resolution, n_folds):
        res_rad = np.deg2rad(resolution)
        k = np.log(1 / 2) / (np.log(np.cos((n_folds / 4 * res_rad)) ** 2))
        return k

    def _get_phi_grid(self):
        y_grid, x_grid = np.mgrid[0 : self.imshape[0], 0 : self.imshape[1]]
        center_row, center_col = self.center
        return np.arctan2(y_grid - center_row, x_grid - center_col)

    def generate_filter(self):
        """Generate the symmetry adapted filter (SAF).

        Returns
        -------
        saf : ndarray
            The generated SAF of shape `imshape`.
        """
        phi_grid = self._get_phi_grid()
        phi_0_rad = np.deg2rad(self.phi_0)
        saf = (
            np.abs(np.cos(0.5 * self.n_folds * (phi_grid - phi_0_rad)))
            ** self.k
        )
        return saf

    def generate_mask(self, width):
        """Generate a mask that covers each lobe out to an angular
        width.

        A pixel is in the mask when its direction from `center` lies within
        `width` / 2 degrees of a lobe center, and a width of 0 masks
        nothing. The SAF falls off steadily
        from each lobe center to halfway between lobes, so the mask equals
        the SAF thresholded at cos(`n_folds` * `width` / 4) ** `k`, but its
        width does not depend on `resolution`.

        Parameters
        ----------
        width : float
            The full angular width of the mask around each lobe, in
            degrees. It must be from 0, which masks nothing, to one period
            of 360 / `n_folds` degrees, which covers every direction.

        Returns
        -------
        mask : ndarray of bool
            The mask of shape `imshape`, True on and near each lobe.

        Raises
        ------
        ValueError
            If `width` is not from 0 to 360 / `n_folds`.
        """
        period = 360 / self.n_folds
        if not 0 <= width <= period:
            raise ValueError(
                "The mask width must be from 0 to one period of 360 / "
                f"n_folds = {period:g} degrees, but {width} was given.\n"
                "Please choose a width from 0, for no masking, to "
                f"{period:g} degrees."
            )
        if width == 0:
            mask = np.zeros(self.imshape, dtype=bool)
            return mask
        phi_grid = self._get_phi_grid()
        phi_0_rad = np.deg2rad(self.phi_0)
        lobe_profile = np.abs(
            np.cos(0.5 * self.n_folds * (phi_grid - phi_0_rad))
        )
        mask = lobe_profile >= np.cos(np.deg2rad(self.n_folds * width / 4))
        return mask
