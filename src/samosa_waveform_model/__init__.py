# -*- coding: utf-8 -*-

"""
python package for the SAMOSA/SAMOSA+ waveform model. Based on SAMPy by CLS
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = [
    "__RESOURCE_DIR__",
    "__LUT_DIR__",
    "SARParameters",
    "SensorParameters",
    "PlatformLocation",
    "ScenarioData",
    "WaveformModelParameters",
    "SAMOSAWaveformModel",
    "SAMOSA_MODEL_TERMS_LUT",
    "ALPHA_POWER_PTR_LUTS",
]

# This needs to be imported at the very beginning to ensure that the resource directory
# is set up correctly before the parsing of the resource files (Platform Presets, Lookup Tables)
from samosa_waveform_model._package import PACKAGE_DIR, RESOURCE_DIR, VERSION
__version__ = VERSION
__PACKAGE_DIR__ = PACKAGE_DIR
__RESOURCE_DIR__ = RESOURCE_DIR
__LUT_DIR__ = RESOURCE_DIR / "lut"

from samosa_waveform_model.enums import WaveformModelEngines
from samosa_waveform_model.datamodels import SensorParameters, SARParameters, PlatformLocation
from samosa_waveform_model.samosaplus import ScenarioData, WaveformModelParameters, SAMOSAWaveformModel
from samosa_waveform_model.lut import SAMOSA_MODEL_TERMS_LUT, ALPHA_POWER_PTR_LUTS
from samosa_waveform_model.presets import SENSORS_PRESETS, ORBIT_EXAMPLE, SurfaceTypeLead, SurfaceTypeSeaIce
