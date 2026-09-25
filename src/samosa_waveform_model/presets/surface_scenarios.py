# -*- coding: utf-8 -*-

"""
These are preset surface scenarios for the SAMOSA waveform model to be used for testing and validation purposes.
They provide predefined parameters for different surface types, such as typical lead and sea ice.
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from samosa_waveform_model.datamodels import WaveformModelParameters


class SurfaceTypeLead(WaveformModelParameters):

    def __init__(
        self,
        significant_wave_height=0.0,
        nu=1e6,
        epoch=0.0
    ):
        super(SurfaceTypeLead, self).__init__(
            significant_wave_height=significant_wave_height,
            nu=nu,
            epoch=epoch
        )


class SurfaceTypeSeaIce(WaveformModelParameters):

    def __init__(
        self,
        significant_wave_height=0.2,
        nu=1000,
        epoch=0.0
    ):
        super(SurfaceTypeSeaIce, self).__init__(
            significant_wave_height=significant_wave_height,
            nu=nu,
            epoch=epoch
        )
