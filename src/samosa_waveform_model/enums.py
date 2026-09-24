# -*- coding: utf-8 -*-

"""
Enumerations used for the package
"""

from enum import StrEnum


class WaveformModelEngines(StrEnum):
    """
    Different implementations of the SAMOSA waveform model/retracker.
    """
    SAMOSA = "samosa"
    SAMOSAPLUS = "samosa+"
