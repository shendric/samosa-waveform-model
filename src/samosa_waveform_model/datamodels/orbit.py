# -*- coding: utf-8 -*-
"""
This package contains the data mode for radar altimeter sensor location and attitude and other
orbit related parameters.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import numpy as np
from functools import cached_property
from dataclasses import dataclass
from typing import Optional

from samosa_waveform_model.constants import CONSTANTS


@dataclass
class PlatformLocation:
    """
    """
    latitude: Optional[float] = None  # latitude in degree for the waveform under iteration
    longitude: Optional[float] = None  # longitude in degree between -180, 180 for the waveform under iteration
    altitude: Optional[float] = None  # Orbit height in meter for the waveform under iteration
    velocity: Optional[float] = None  # Satellite Velocity in m/s
    height_rate: Optional[float] = None  # Orbit Height rate in m/s for the waveform under iteration
    pitch: Optional[float] = None  # Altimeter Reference Frame Pitch in radian
    roll: Optional[float] = None  # Altimeter Reference Frame Roll in radian
    track_sign: int = 0  # -1 for ascending & +1 for descending, set it to zero if flag_slope=False in

    @cached_property
    def earth_radius(self) -> float:
        return np.sqrt(
            CONSTANTS.R_e ** 2.0 * (np.cos(np.deg2rad(self.latitude))) ** 2. +
            CONSTANTS.b_e ** 2.0 * (np.sin(np.deg2rad(self.latitude))) ** 2.
        )

    @cached_property
    def kappa(self) -> float:
        return 1. + self.altitude / self.earth_radius

    @cached_property
    def orbit_slope(self) -> float:
        return self.track_sign * \
               ((CONSTANTS.R_e ** 2 - CONSTANTS.b_e ** 2) / (2. * self.earth_radius ** 2)) * \
               np.sin(np.deg2rad(2. * self.latitude)) - \
               (-self.height_rate / self.velocity)