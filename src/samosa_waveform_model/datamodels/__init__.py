# -*- coding: utf-8 -*-
"""
This module contains the data models (dataclasses and pydantic Basemodels with extended
validation) for the SAMOSA waveform model package.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = [
    "model",
    "sensor",
    "orbit",
    "PlatformLocation",
    "SARParameters",
    "SensorParameters",
    "ScenarioData"
]


from typing import Optional, Literal, Tuple, Dict
import numpy as np

from samosa_waveform_model.constants import CONSTANTS
from samosa_waveform_model.enums import WaveformModelEngines
from samosa_waveform_model.datamodels.sensor import SensorParameters, SARParameters
from samosa_waveform_model.datamodels.orbit import PlatformLocation
from samosa_waveform_model.lut import ALPHA_POWER_PTR_LUTS



