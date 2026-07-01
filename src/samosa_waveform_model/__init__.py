# -*- coding: utf-8 -*-

"""
python package for the SAMOSA+  conversion. Based on sampy by CLS
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import warnings
from samosa_waveform_model.dataclasses import SensorParameters, SARParameters, PlatformLocation, WaveformModelParameters
from samosa_waveform_model.model import ScenarioData, SAMOSAWaveformModel
from samosa_waveform_model.lut import CS2_LOOKUP_TABLES

from samosa_waveform_model.model_cy import SAMOSAWaveformModel as SAMOSAWaveformModelCython

try:
    from samosa_waveform_model.model_cy import SAMOSAWaveformModel as SAMOSAWaveformModelCython
except ImportError as ie:
    warnings.warn(f"Could not import SAMOSAWaveformModelCython: {ie}")
    SAMOSAWaveformModelCython = None

__all__ = [
    "SARParameters",
    "SensorParameters",
    "PlatformLocation",
    "ScenarioData",
    "WaveformModelParameters",
    "SAMOSAWaveformModel",
    "SAMOSAWaveformModelCython",
    "CS2_LOOKUP_TABLES",
    "scenarios"
]
