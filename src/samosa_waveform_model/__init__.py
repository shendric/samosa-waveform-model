# -*- coding: utf-8 -*-

"""
python package for the SAMOSA/SAMOSA+ waveform model. Based on SAMPy by CLS
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = [
    "SARParameters",
    "SensorParameters",
    "PlatformLocation",
    "ScenarioData",
    "WaveformModelParameters",
    "SAMOSAWaveformModel",
    "SAMOSAModelTermsTable",
    "AlphaPowerPTRTable",
    "scenarios"
]

from samosa_waveform_model.dataclasses import SensorParameters, SARParameters, PlatformLocation, WaveformModelParameters
from samosa_waveform_model.model import ScenarioData, SAMOSAWaveformModel
from samosa_waveform_model.lut import SAMOSAModelTermsTable, AlphaPowerPTRTable
