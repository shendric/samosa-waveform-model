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

from pathlib import Path
from typing import Literal, Optional, Tuple

from functools import cached_property
import numpy as np

from samosa_waveform_model.dataclasses import CONSTANTS


# Store the path to the lookup tables
__LUT_PATH__ = Path(__file__).parent / "lut"

from fsspec.implementations import cached


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


SAMOSA_MODEL_TERMS_LUT = SAMOSAModelTermsTable.from_package_luts()


class AlphaPowerPTRTable(object):
    """
    The class for the alphaPower lookup table, which is used to compute the alphaPower term of the SAMOSA model.
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

    @classmethod
    def from_file(
            cls,
            platform: Optional[str] = None,
            hamming: Optional[bool] = None
    ) -> "AlphaPowerPTRTable":
        """
        Load the lookup table from a file

        :param platform: Optional platform name for the lookup table.
            Needed to construct the expected filename.
        :param hamming: Optional boolean indicating if the lookup table is for hamming or no-hamming windowing.
            Needed to construct the expected filename.

        :raises FileNotFoundError: If the lookup table file does not exist

        :return: AlphaPowerPTRTable object
        """
        expected_filename = __LUT_PATH__ / f"{platform}" / f"alphaPower_table_{platform}_{'hamming' if hamming else 'nohamming'}.csv"
        if not expected_filename.is_file():
            raise FileNotFoundError(f"Lookup table file not found: {expected_filename}")
        data = np.genfromtxt(expected_filename, comments='#', delimiter=',')
        swh = data[:, 0]
        alpha_power = data[:, 1]
        return cls(swh, alpha_power, platform=platform, hamming=hamming)

    def get(self, swh_value: float) -> float:
        breakpoint()


# Deprecated
class SAMOSALookupTables(object):
    """
    Container for SAMOSA lookup tables
    # TODO: Currently hard-coded to CryoSat-2 ICE L1b files -> create more LUT files and make dependent on sensor/mode
    """

    def __init__(self) -> None:
        """
        Load the SAMOSA lookup tables
        """

        lut_folder = Path(__file__).parent / "lut"
        kwargs = dict(dtype='float', comments='#', delimiter=None)
        self.f0 = np.genfromtxt(lut_folder / "LUT_F0.txt", **kwargs)
        self.f1 = np.genfromtxt(lut_folder / "LUT_F1.txt", **kwargs)

        kwargs = dict(dtype='float', comments='#', delimiter=',')
        self.alphap_noweight = np.genfromtxt(lut_folder / self.alphap_noweight_file, **kwargs)
        self.alphap_weight = np.genfromtxt(lut_folder / self.alphap_weight_file, **kwargs)
        self.alphapower_noweight = np.genfromtxt(lut_folder / self.alphapower_noweight_file, **kwargs)
        self.alphapower_weight = np.genfromtxt(lut_folder / self.alphapower_weight_file, **kwargs)

    @property
    def alphap_noweight_file(self) -> str:
        return 'alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_NOHAMMING).txt'

    @property
    def alphap_weight_file(self) -> str:
        return 'alphap_table_DX3000_ZP20_SWH20_10_Sept_2019(CS2_HAMMING).txt'

    # TODO: Check if this is correct (both alphapower weight and noweight point to NOHAMMING)
    @property
    def alphapower_weight_file(self) -> str:
        return 'alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'

    @property
    def alphapower_noweight_file(self) -> str:
        return 'alphaPower_table_CONSTANT_SWH20_10_Feb_2020(CS2_NOHAMMING).txt'


# CS2_LOOKUP_TABLES = SAMOSALookupTables()
