# -*- coding: utf-8 -*-
"""
This module contains preset scenarios for the SAMOSA waveform model,
including predefined surface, sensor orbit scenarios.

These presets can be used for testing and tutorials.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = [
    "sensor_presets.py",
    "surface_presets.py",
    "orbit_presets.py",
    "ORBIT_EXAMPLE",
    "SENSORS_PRESETS",
    "SurfaceTypeLead",
    "SurfaceTypeSeaIce"
]

from samosa_waveform_model.presets.orbit_presets import ORBIT_EXAMPLE
from samosa_waveform_model.presets.sensor_presets import SENSORS_PRESETS
from samosa_waveform_model.presets.surface_presets import SurfaceTypeLead, SurfaceTypeSeaIce

#
# @classmethod
#     def cryosat2_sar_example(
#             cls,
#             loc_parameters: Optional[Dict] = None
#     ):
#         """ Real life CryoSat-2 lead example """
#
#         loc_parameters = {} if loc_parameters is None else loc_parameters
#
#         # Radar altimeter parameters
#         sp, sar = SENSORS_PRESETS.get_presets("cryosat2", "sar")
#         # Example position/attitude
#         # example_loc = dict(latitude=83.9625006,
#         #                    longitude=27.407605,
#         #                    altitude=728518.615,
#         #                    height_rate=0.466,
#         #                    pitch=-0.0010057948506807881,
#         #                    roll=-0.0015263707160146328,
#         #                    velocity=7518.711587141643)
#
#         loc_dict = dict(
#             latitude=83.9625006,
#             longitude=27.407605,
#             altitude=728518.615,
#             height_rate=0.,
#             pitch=0.,
#             roll=0.,
#             velocity=7518.711587141643
#         )
#         loc_dict.update(loc_parameters)
#
#         geo = PlatformLocation(**loc_dict)
#         sar = SARParameters()
#         sar.compute_multi_look_parameters(geo=geo, sp=sp)
#
#         return cls(sp, geo, sar)