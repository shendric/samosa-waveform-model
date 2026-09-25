# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["ORBIT_EXAMPLE"]
from samosa_waveform_model.datamodels import PlatformLocation


ORBIT_EXAMPLE = PlatformLocation(
    latitude=83.9625006,
    longitude=27.407605,
    altitude=728518.615,
    height_rate=0.,
    pitch=0.,
    roll=0.,
    velocity=7518.711587141643
)
