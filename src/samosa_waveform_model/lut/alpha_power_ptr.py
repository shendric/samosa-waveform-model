# -*- coding: utf-8 -*-
"""
This module contains the lookup tables for the significant wave height (swh) dependent alpha power term
of the SAMOSA waveform model.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from parse import parse
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
from scipy.interpolate import interp1d

from samosa_waveform_model import __LUT_DIR__


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
    different presets and hamming options.

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
        lut_files: List[Path] = sorted(__LUT_DIR__.rglob("alpha_power_ptr*.csv"))
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

    def get(self, platform: str, swh: float, hamming: bool) -> float:
        """
        Get the alphaPower lookup table for the specified platform and hamming option.

        :param swh: significant wave height for which the alpha power value is to be retrieved
        :param platform: The platform name (e.g., "cryosat2) defined in the lookup table filename.
        :param hamming: Boolean indicating if the lookup table is for hamming or no-hamming windowing.

        :return: AlphaPowerPTRTable object
        """
        lut_table = self.lut_tables.get((platform, hamming))
        err_msg = f"No lookup table found for platform {platform=} and hamming {hamming=} [{list(self.lut_tables.keys())}]"
        assert lut_table is not None, err_msg

        return lut_table.get(swh)
