# -*- coding: utf-8 -*-

"""
This module contains the lookup tables for the SAMOSA model, for both the SAMOSA SAR Model Term of zero and first-order
as well as sensor-specific alpha power (PTR) lookup tables.

It is expected that the lookup tables are stored in the "lut" folder of the package.

All lookup tables are loaded at the package initialization and can be accessed via the global variables
`SAMOSA_MODEL_TERMS_LUT` and `ALPHA_POWER_PTR_LUT[platform]`. This is done to ensure that the lookup tables
are not loaded multiple times for repeated calls to the waveform model during waveform fitting.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from parse import parse
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from functools import cached_property
import numpy as np
from scipy.interpolate import interp1d

from samosa_waveform_model.dataclasses import CONSTANTS


# Store the path to the lookup tables
__LUT_PATH__ = Path(__file__).parent / "lut"


class SAMOSAModelTermsTable(object):
    """
    The class for the zero (f_0) and first order terms (f_1) of the SAMOSA model, which are stored in lookup tables.
    """

    def __init__(
            self,
            xi: np.ndarray,
            f0: np.ndarray,
            f1: np.ndarray
    ) -> None:
        """
        Stores the lookup tables for the zero (f_0) and first order terms (f_1) of the SAMOSA model
        depended on the variable xi (Described as "generic independent variable" in Dinardo, 2020).
        and very likely the variable `csi` in the original SAMPy code

        :param xi: The independent variable
        :param f0: The zero order term
        :param f1: The first order term
        """
        self.xi_lut = xi
        self.f0_lut = f0
        self.f1_lut = f1

    @classmethod
    def from_package_luts(cls) -> "SAMOSAModelTermsTable":
        """
        Load the SAMOSA model terms lookup tables from the package's lut folder.

        The lookup table files contains two columns (xi, f0) and (xi, f1) respectively.
        The first column is the independent variable xi, which is expected to be identical
        in the two files.

        :return: SAMOSAModelTermsLUT object
        """
        lut_folder = Path(__file__).parent / "lut"
        kwargs = dict(dtype='float', comments='#', delimiter=None)
        f0 = np.genfromtxt(lut_folder / "LUT_F0.txt", **kwargs)
        f1 = np.genfromtxt(lut_folder / "LUT_F1.txt", **kwargs)
        assert np.array_equal(f0[:, 0], f1[:, 0]), "LUT_F0 and LUT_F1 must share the same xi values in the first column"
        xi = f0[:, 0]
        f0 = f0[:, 1]
        f1 = f1[:, 1]
        return cls(xi, f0, f1)

    def get(
            self,
            order: Literal[0, 1],
            xi: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:

        match order:
            case 0: return self._get_f0(xi, z)
            case 1: return self._get_f1(xi, z)
            case _: raise ValueError(f"Invalid order: {order}. Must be 0 or 1.")

    def _get_f0(self, xi: np.ndarray, z: np.ndarray) -> np.ndarray:
        f0 = self._get_clipped_fi(0, xi)
        f0 = self._set_f0_z_idx_max_f0(xi, f0, z)
        f0 = self._set_fi_csi_eq_0(f0, xi, CONSTANTS.f0_csi_0)
        f0 = self._set_fi_csi_lt_csi_min(f0, xi)
        return f0

    def _get_f1(self, xi: np.ndarray, z: np.ndarray) -> np.ndarray:
        f1 = self._get_clipped_fi(1, xi)
        f1 = self._set_f1_csi_gt_csi_max(f1, xi, z)
        f1 = self._set_fi_csi_eq_0(f1, xi, CONSTANTS.f1_csi_0)
        f1 = self._set_fi_csi_lt_csi_min(f1, xi)
        return f1

    def _get_clipped_fi(self, order: Literal[0, 1], xi: np.ndarray) -> np.ndarray:
        """
        Get the clipped values of f0 or f1 based on the provided xi values.

        :param xi: The independent variable
        :return: Clipped values of f0 or f1
        """
        lut_xi, lut_fi = self._get_lut(order)
        xi_min, xi_max = self.xi_range
        fi = np.zeros(np.shape(xi))
        clip_fi = np.bitwise_and(xi >= xi_min, xi <= xi_max)
        idx = np.floor((lut_xi.size - 1) * ((xi[clip_fi] - xi_min) / (xi_max - xi_min))).astype(int)
        fi[clip_fi] = (xi[clip_fi] - lut_xi[idx]) * ((lut_fi[idx + 1] - lut_fi[idx]) / (
                lut_xi[idx + 1] - lut_xi[idx])) + lut_fi[idx]
        return fi

    def _set_f0_z_idx_max_f0(
            self,
            xi: np.ndarray,
            f0: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        idx_max_f0 = np.where(xi > self.xi_range[1])
        z_idx_max_f0 = z[idx_max_f0]
        f0[idx_max_f0] = 1. / 2. * CONSTANTS.sqrt_pi / z_idx_max_f0 ** (1. / 4.) * (
                1. + 3. / (32. * z_idx_max_f0) + 105. / (
                2048. * z_idx_max_f0 ** 2) + 10395. / (
                        196608. * z_idx_max_f0 ** 3))
        return f0

    def _set_f1_csi_gt_csi_max(
            self,
            f1: np.ndarray,
            xi: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        idx_max_f1 = np.where(xi > self.xi_range[1])
        f1[idx_max_f1] = (1. / 2.) * 1. / 4. * CONSTANTS.sqrt_pi / (z[idx_max_f1]) ** (3. / 4.)
        return f1

    @staticmethod
    def _set_fi_csi_eq_0(f0: np.ndarray, csi: np.ndarray, value: float) -> np.ndarray:
        f0[np.where(csi == 0)] = value
        return f0

    def _set_fi_csi_lt_csi_min(self, f0, xi):
        f0[np.where(xi < self.xi_range[0])] = 0
        return f0

    def _get_lut(self, order: Literal[0, 1]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the lookup table for the specified order (0 or 1).

        :param order: The order of the term (0 for f0, 1 for f1)

        :return: Tuple of (xi_lut, fi_lut)
        """
        match order:
            case 0: return self.xi_lut, self.f0_lut
            case 1: return self.xi_lut, self.f1_lut
            case _: raise ValueError(f"Invalid order: {order}. Must be 0 or 1.")

    @cached_property
    def xi_range(self) -> Tuple[float, float]:
        """
        Return the min and max value of xi

        :return: Min(xi), Max(xi)
        """
        return np.min(self.xi_lut), np.max(self.xi_lut)


class AlphaPowerPTRTable(object):
    """
    The class for the alphaPower lookup table, which is used to compute the alphaPower term of the SAMOSA model.
    The content of the
    """

    def __init__(
            self,
            swh: np.ndarray,
            alpha_power: np.ndarray,
            platform: Optional[str] = None,
            hamming: Optional[bool] = None
    ) -> None:
        """
        Stores the lookup table for the alphaPower term of the SAMOSA model depended on the significant wave height (swh).

        :param swh: The significant wave height
        :param alpha_power: The alphaPower term
        """
        self.platform = platform
        self.hamming = hamming
        self.swh = swh
        self.alpha_power = alpha_power

        # Construct an interpolation function for the alpha power term that can be called
        # later to get the alpha power value for a given significant wave height (swh).
        # If the swh value is outside the range of the lookup table, the function will return the
        # alpha power value at the closest boundary (min or max swh).
        fill_values = (self.swh[0], self.swh[-1])
        interp_kwargs = dict(kind='linear', bounds_error=False, fill_value=fill_values)
        self.interp_func = interp1d(self.swh, self.alpha_power, **interp_kwargs)

    @classmethod
    def from_file(
            cls,
            filename: Union[str, Path],
            platform: Optional[str] = None,
            hamming: Optional[bool] = None
    ) -> "AlphaPowerPTRTable":
        """
        Load the lookup table from a csv file

        :param filename: Path to the lookup table file. The file is expected to be in CSV format
            with two columns: swh, alpha_power.
        :param platform: Optional platform name for the lookup table.
            Needed to construct the expected filename.
        :param hamming: Optional boolean indicating if the lookup table is for hamming or no-hamming windowing.
            Needed to construct the expected filename.

        :raises FileNotFoundError: If the lookup table file does not exist

        :return: AlphaPowerPTRTable object
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Lookup table file not found: {filename}")

        # csv format with two columns: swh, alpha_power
        data = np.genfromtxt(filename, skip_header=1, comments='#', delimiter=',')
        swh = data[:, 0]
        alpha_power = data[:, 1]
        return cls(swh, alpha_power, platform=platform, hamming=hamming)

    def get(self, swh_value: float) -> float:
        """
        Get the alphaPower value for the specified significant wave height (swh_value) by interpolating the lookup table.

        :param swh_value: The value for which

        :return: alphaPower value corresponding to the specified significant wave height
        """
        return self.interp_func(swh_value)



class AlphaPowerPTRTableCatalogue(object):
    """
    Class to store alpha_power_ptr lookup tables as function of significant wave height for
    different platforms and hamming options.

    :param lut_tables: A dictionary mapping (platform, hamming) tuples to AlphaPowerPTRTable objects.
    """

    def __init__(self, lut_tables: dict[Tuple[str, bool], AlphaPowerPTRTable]) -> None:
        self.lut_tables = lut_tables

    @classmethod
    def from_package(cls) -> "AlphaPowerPTRTableCatalogue":
        """
        Read the alphaPower lookup tables from the package's lut folder and
        return a catalogue of the available tables.

        NOTE: The lookup tables are expected to be stored in the "lut/alpha_power_ptr" folder of the package.
              with a filenaming convention of "alphaPower_table_{platform}_{hamming}.csv"
              where {platform} is the platform name and {hamming} is either "hamming" or "nohamming".

        :return: AlphaPowerPTRTableCatalogue object
        """

        # Find all available alpha_power_ptr files in the lut subfolder
        lut_files = sorted(__LUT_PATH__.rglob("alpha_power_ptr*.csv"))
        lut_table_dict = {}

        for lut_file in lut_files:

            # Extract platform and hamming from the filename
            result = parse("alpha_power_ptr_{platform}_{hamming}", lut_file.stem)
            assert result is not None, f"Invalid filename format: {lut_file.name}"
            platform, hamming_str = result["platform"], result["hamming"]

            assert hamming_str in ["hamming", "nohamming"], f"Invalid hamming value in filename: {lut_file.name}"
            hamming = {"hamming": True, "nohamming": False}[hamming_str]

            lut_table_dict[(platform, hamming)] = AlphaPowerPTRTable.from_file(
                lut_file, platform=platform, hamming=hamming
            )

        return cls(lut_table_dict)

    def get(self, platform: str, swh: float, hamming: bool) -> "AlphaPowerPTRTable":
        """
        Get the alphaPower lookup table for the specified platform and hamming option.

        :param platform: The platform name (e.g., "cryosat2", "sentinel3a", "sentinel3b")
        :param hamming: Boolean indicating if the lookup table is for hamming or no-hamming windowing.

        :return: AlphaPowerPTRTable object
        """

        # Get the lookup table for the specified platform and hamming option
        lut_table = self.lut_tables.get((platform, hamming))
        err_msg = f"No lookup table found for platform {platform=} and hamming {hamming=} [{list(self.lut_tables.keys())}]"
        assert lut_table is not None, err_msg

        return lut_table.get(swh)

        breakpoint()
        # return AlphaPowerPTRTable.from_file(platform=platform, hamming=hamming)


# Read all available lookup tables from the package's lut folder
# and store them in global variables
# NOTE: This needs to be done at package initialization
SAMOSA_MODEL_TERMS_LUT = SAMOSAModelTermsTable.from_package_luts()
ALPHA_POWER_PTR_LUTS = AlphaPowerPTRTableCatalogue.from_package()