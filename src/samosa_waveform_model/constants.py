# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["CONSTANTS"]

import numpy as np
from functools import cached_property
from dataclasses import dataclass


@dataclass
class Constants:
    """ A set of constants needed for the waveform model """
    c0: float = 299792458.  # speed of light in m/sec
    R_e: float = 6378137.  # Reference Ellipsoid Earth Radius in m
    f_e: float = 1. / 298.257223563  # Reference Ellipsoid Earth Flatness

    @cached_property
    def ecc_e(self) -> float:
        return np.sqrt((2. - self.f_e) * self.f_e)

    @cached_property
    def b_e(self) -> float:
        return self.R_e * np.sqrt(1. - self.ecc_e ** 2.)


CONSTANTS = Constants()
