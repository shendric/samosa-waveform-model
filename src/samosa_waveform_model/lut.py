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
from typing import Literal
import numpy as np


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
        self.xi = xi
        self.f0 = f0
        self.f1 = f1

    @classmethod
    def from_package_luts(cls) -> "SAMOSAModelTermsTable":
        """
        Load the SAMOSA model terms lookup tables from the package's lut folder.

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
            clip_xi_range: bool = True,
            constant_xi0: bool = True,
    ) -> np.ndarray:
        breakpoint()


SAMOSA_MODEL_TERMS_LUT = SAMOSAModelTermsTable.from_package_luts()


class AlphaPowerPTRLUT(object):
    """
    The class for the alphaPower lookup table, which is used to compute the alphaPower term of the SAMOSA model.
    """

    def __init__(
            self,
            swh: np.ndarray,
            alpha_power: np.ndarray
    ) -> None:
        """
        Stores the lookup table for the alphaPower term of the SAMOSA model depended on the significant wave height (swh).

        :param swh: The significant wave height
        :param alpha_power: The alphaPower term
        """
        self.swh = swh
        self.alpha_power = alpha_power

    @classmethod
    def from_file(cls, lut_file: Path) -> "AlphaPowerPTRLUT":
        """
        Load the lookup table from a file
        :param lut_file: Path to the lookup table file
        :return: AlphaPowerPTRLUT object
        """
        data = np.genfromtxt(lut_file, comments='#', delimiter=',')
        swh = data[:, 0]
        alpha_power = data[:, 1]
        return cls(swh, alpha_power)

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
