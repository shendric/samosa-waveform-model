# -*- coding: utf-8 -*-

"""
This module contains the lookup tabless that are read from files during initialization.
These can be accessed via the global variables `SAMOSA_MODEL_TERMS_LUT` and `ALPHA_POWER_PTR_LUT[platform]`. T
The loading of tables upon package initialization ensures that the lookup tables
are not loaded multiple times for repeated calls to the waveform model during waveform fitting.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"
__all__ = ["SAMOSA_MODEL_TERMS_LUT", "ALPHA_POWER_PTR_LUTS"]

from samosa_waveform_model.lut.alpha_power_ptr import AlphaPowerPTRTableCatalogue
from samosa_waveform_model.lut.model_terms import SAMOSAModelTermsTable

SAMOSA_MODEL_TERMS_LUT = SAMOSAModelTermsTable.from_package_luts()
ALPHA_POWER_PTR_LUTS = AlphaPowerPTRTableCatalogue.from_package()