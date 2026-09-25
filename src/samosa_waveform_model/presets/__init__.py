# -*- coding: utf-8 -*-
"""
This module contains preset scenarios for the SAMOSA waveform model,
including predefined surface, sensor orbit scenarios.

These presets can be used for testing and tutorials.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = [
    "get_scenario_preset",
    "sensor_presets",
    "surface_presets",
    "orbit_presets",
    "ORBIT_EXAMPLE",
    "SENSORS_PRESETS",
    "SurfaceTypeLead",
    "SurfaceTypeSeaIce"

]

from typing import Optional

from samosa_waveform_model.samosaplus import ScenarioData
from samosa_waveform_model.datamodels import PlatformLocation
from samosa_waveform_model.presets.orbit_presets import ORBIT_EXAMPLE
from samosa_waveform_model.presets.sensor_presets import SENSORS_PRESETS
from samosa_waveform_model.presets.surface_presets import SurfaceTypeLead, SurfaceTypeSeaIce



def get_scenario_preset(
        platform: str,
        mode: str,
        geo_params: Optional[PlatformLocation] = None
) -> ScenarioData:
    """
    Retrieve a preset scenario for a given platform and radar mode and
    a default orbit scenario (which can be overridden by providing a custom PlatformLocation object).

    :param platform: The platform name (e.g., "cryosat2").
    :param mode: The radar mode name (e.g., "sar").
    :param geo_params: The geographic parameters (e.g., platform location).
        If None, the default ORBIT_EXAMPLE will be used.

    :return: The ScenarioData input data model for the SAMOSA waveform model.
    """
    geo_params = geo_params if isinstance(geo_params, PlatformLocation) else ORBIT_EXAMPLE
    sensor_params, sar_params = SENSORS_PRESETS.get_preset(platform, mode)
    sar_params.compute_multi_look_parameters(geo_params, sensor_params)
    return ScenarioData(sensor_params, geo_params, sar_params)
