# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import warnings
from warnings import warn
import bottleneck as bn
import pandas as pd
import numpy as np

from pydantic import BaseModel, PositiveInt
from typing import Dict, Optional, Tuple

from samosa_waveform_model.enums import WaveformModelEngines
from samosa_waveform_model.dataclasses import (SensorParameters, PlatformLocation, SARParameters,
                                               CONSTANTS, WaveformModelOutput, WaveformModelParameters)
from samosa_waveform_model.lut import CS2_LOOKUP_TABLES


from samosa_waveform_model.funcs_py import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
                                            ddm_mask_ranges)


# try:
#     from samosa_waveform_model.funcs import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
#                                             ddm_mask_ranges)
# except ImportError:
#     msg = """
#     Could not import the compiled functions for the SAMOSA+ waveform model.
#     Please install the compiled functions module func by running
#     'python setup.py build_ext --inplace
#     -> Using python implementation instead
#     """
#     warnings.warn(msg)
#     from samosa_waveform_model.funcs_py import (compute_gl, compute_gamma0, compute_t_kappa, compute_f0, compute_f1,
#                                             ddm_mask_ranges)


class ScenarioData(object):
    """ Scenario (radar parameters, pl for running the SAMOSA+ waveform model """

    def __init__(
            self,
            rp: "SensorParameters",
            geo: "PlatformLocation",
            sar: "SARParameters"

    ) -> None:
        """
        Bundle of all input parameters for the SAMOSA+ waveform model. .

        :param rp: Sensor (Radar) parameters
        :param geo: Location and attitude of the platform (satellite)
        :param sar: SAR processing parameters
        """
        self.rp = rp
        self.geo = geo
        self.sar = sar

    @classmethod
    def cryosat2_sar_example(
            cls,
            loc_parameters: Optional[Dict] = None
    ):
        """ Real life CryoSat-2 lead example """

        loc_parameters = {} if loc_parameters is None else loc_parameters

        # Radar altimeter parameters
        sp = SensorParameters.cryosat2_sar()
        # Example position/attitude
        # example_loc = dict(latitude=83.9625006,
        #                    longitude=27.407605,
        #                    altitude=728518.615,
        #                    height_rate=0.466,
        #                    pitch=-0.0010057948506807881,
        #                    roll=-0.0015263707160146328,
        #                    velocity=7518.711587141643)

        loc_dict = dict(
            latitude=83.9625006,
            longitude=27.407605,
            altitude=728518.615,
            height_rate=0.,
            pitch=0.,
            roll=0.,
            velocity=7518.711587141643
        )
        loc_dict.update(loc_parameters)

        geo = PlatformLocation(**loc_dict)
        sar = SARParameters()
        sar.compute_multi_look_parameters(geo=geo, sp=sp)

        return cls(sp, geo, sar)

    def get_alpha_power(
            self,
            engine: WaveformModelEngines,
            swh: Optional[float] = None
    ) -> Tuple[float, float]:
        """

        # TODO: A lot

        :param engine:
        :param swh:

        :return:
        """

        if self.sar.hamming_weighting:
            match engine:
                case WaveformModelEngines.SAMOSA:
                    ind = bn.nanargmin(abs(self.lut.alphap_weight[:, 0] - swh))
                    alpha_p = self.lut.alphap_weight[:, 1][ind]
                    alpha_power = 0.47356
                case WaveformModelEngines.SAMOSAPLUS:
                    alpha_p = 0.42349
                    alpha_power = 0.47356
        else:
            alpha_p, alpha_power = self.get_alpha_power_no_weights(swh)
        return alpha_p, alpha_power

    def get_alpha_power_no_weights(self, swh):
        # TODO: To be confirmed (and renamed) that weights means Hamming weighting
        ind = np.argmin(abs(self.lut.alphap_noweight[:, 0] - swh))
        alpha_p = self.lut.alphapower_noweight[:, 1][ind]

        ind = np.argmin(abs(self.lut.alphapower_noweight[:, 0] - swh))
        alpha_power = self.lut.alphapower_noweight[:, 1][ind]
        return alpha_p, alpha_power


class FixedScenarioVariables(object):

    def __init__(
                self,
                engine: WaveformModelEngines,
                scenario: ScenarioData,
                use_slope: bool = False,
    ) -> None:
            """
            A class for the pre-computation of fixed variables for the SAMOSA+ waveform model.

            This class exists for the purpose of fitting the waveform model to data,
            where the fixed variables can be pre-computed once and don't need to be computed
            in every iteration of the fitting process.

            :param engine: The waveform model engine to use (SAMOSA or SAMOSA+)
            :param scenario: The waveform model input data (sensor parameters, platform location, SAR parameters)
            :param use_slope:
            """
            pass
            # self.scenario = scenario
            # self.flag_slope = int(use_slope)
            # self.mask_ranges = mask_ranges
            # self.lut = CS2_LOOKUP_TABLES  # TODO: Move to scenario data (specifically radar parameters)
            # self.static_parameters = {}
            # self._precompute_static_parameters()


class SAMOSAFitHouseKeeping(object):

    def __init__(
            self,
            collect_fit_params: bool = False
    ) -> None:
        """
        A class for the housekeeping of the SAMOSA+ waveform model fitting process.

        This class exists for the purpose of fitting the waveform model to data,
        where the fit parameters can be collected and stored for later analysis.

        :param collect_fit_params: Whether to collect fit parameters during the fitting process.
        """
        self.collect_fit_params = collect_fit_params
        self.fit_params = []
        self.generate_ddm_counter = 0

    def append(self, waveform_model_parameters: "WaveformModelParameters") -> None:
        """
        Append the fit parameters to the list of fit parameters.

        :param waveform_model_parameters: The fit parameters to append.
        """
        self.fit_params.append(waveform_model_parameters)
        self.generate_ddm_counter += 1


class SAMOSAWaveforModelConfig(BaseModel):
    """
    A class for the configuration of the SAMOSA+ waveform model.
    """
    use_slope: bool = False
    beamsamp_factor: PositiveInt = 1
    norm_model_power: bool = True

    @property
    def flag_slope(self) -> int:
        return int(self.use_slope)


class SAMOSAWaveformModel(object):
    """
    A class for the modeling of waveforms using the SAMOSA+ waveform model
    (Currently ocean waveforms with free parameter of range and significant waveheight only)
    """

    def __init__(
            self,
            engine: WaveformModelEngines,
            scenario: ScenarioData,
            use_slope: bool = False,
            beamsamp_factor: PositiveInt = 1,
            norm_model_power: bool = True,
            collect_fit_params: bool = False
    ) -> None:
        """
        Initialize the forward model

        :param engine:
        :param scenario:
        :param use_slope:
        """

        # Store the input parameters
        self.engine = engine
        self.scenario = scenario
        self.cfg = SAMOSAWaveforModelConfig(
            use_slope=use_slope,
            beamsamp_factor=beamsamp_factor,
            norm_model_power=norm_model_power
        )

        # Pre-compute the parameters that are independent of the waveform model parameters (SWH, MSS, epoch)
        # NOTE: This is done for efficiency during repeated computations of the waveform model
        #       with the same scenario (e.g. during fitting)
        self.static_parameters = FixedScenarioVariables(
            engine=engine,
            scenario=scenario,
            use_slope=use_slope
        )

        # Housekeeping variables
        # NOTE: These are only used to store fitting parameters
        #       used for each waveform model computation for each iteration
        #       and are not required for the waveform model itself
        self.fit_params = SAMOSAFitHouseKeeping(collect_fit_params)

    def generate_delay_doppler_waveform(
            self,
            waveform_model_parameters: "WaveformModelParameters",
    ) -> "WaveformModelOutput":
        """
        Compute a delay doppler waveform. This is derived from sampy.SAMOSA.__Generate_SamosaDDM

        :param waveform_model_parameters:

        :return:
        """

        # Create short variables name for brevity of some parameters
        geo = self.scenario.geo
        rp = self.scenario.rp
        wfm = waveform_model_parameters
        lut = self.lut
        swh = wfm.significant_wave_height
        # nu = 1. / wfm.mean_square_slope
        nu = wfm.nu
        alt = geo.altitude
        tau = self.scenario.rp.tau - wfm.epoch
        beam_index = self.scenario.sar.beam_index

        # --- Collect the waveform model parameters if requested --->
        # This is useful to restore the parameters variations
        # in an optimization process
        if self.fit_params.collect_fit_params:
            self.fit_params.append(wfm)

        # --- Compute variables independent of waveform model parameters --->
        # NOTE: For repeated computations, these all need to be computed once
        p = self.static_parameters

        dk = (tau * rp.bandwidth)
        yk = 0 * dk
        dk_positive = np.where(dk > 0)
        yk[dk_positive] = p["Ly"] * np.sqrt(dk[dk_positive])

        sigma_s = (swh / (4. * p["Lz"]))

        # surface elevation standard deviation
        sigma_z = (swh / 4.)


        """
        Notes to the use of `alpha_p` and `alpha_power`: 
        
        In the formulas in Dinardo 2020, there is just on alpha_p. Here, in the code
        the alpha_p goes into the computation of gl and alpha_power goes into
        the scaling factor for delay doppler map.
        
        The reason for the two alpha_power factor could be that one is
        for the range and one for the azimuth PTR.
        
        The parameter `alpha_p` goes into compute_gl and `alpha_power` goes into
        the constant factor for the delay doppler map. 
        
        According to a code comment in pysamosa (https://pypi.org/project/pysamosa/), 
        `alpha_power` is an average and constant values which should be used for the 
        delay dopper map scaling factor (thus the alpha_power lookup table in SAMPy
        only includes a singular alpha power value and is the same for Hamming and no Hamming).
        
        `alpha_p`instead may vary as function of significant waveheight. 
        But when zero-padding is applied, than alpha_p may also be constant 
        value (section 3.2.3 in Dinardo et al. 2020, with a fixed value of 0.55). 
        Nevertheless, SAMPy sets `alpha_p` to 0.42349 for SAMOSA+. 
        But since the SAMOSA+ retracker uses SAMOSA (with nu set to zero) for the 
        significant waveheight step, an lookup table for `alpha_p` is required 
        for both Hamming and no-Hamming configurations. 
        """
        alpha_p, alpha_power = self.get_alpha_power(swh)

        gl = compute_gl(alpha_p, p["Lx"], p["Ly"], p["Lz"], beam_index, p["ls"], swh)

        csi = gl[None, :] * dk[:, None]
        z = 1. / 4. * csi ** 2

        # gamma0: Surface backscatter response
        gamma_0 = compute_gamma0(p["alpha_y"], p["yp"], p["alpha_x"], nu, alt, p["xl"], p["xp"], yk)

        # Equation 3.19 in Dinardo
        t_kappa = compute_t_kappa(z, dk, nu, alt, p["alpha_y"], p["yp"], p["Ly"])

        # f0 : zero order term of the SAMOSA SAR return waveform model
        f0 = compute_f0(csi, p["csi_min_F0"], p["csi_max_F0"], z, lut)

        # f1 : first order term of the SAMOSA SAR return waveform model
        f1 = compute_f1(csi, p["csi_min_F1"], p["csi_max_F1"], z, lut)

        f = (f0 + sigma_z / p["Lg"] * t_kappa * gl * sigma_s * f1)

        # ddm: delay doppler map
        # TODO: Where is **4 in the const coming from? It is **2 in eq 3.15 in Dinardo 2020
        const = np.sqrt(2. * np.pi * alpha_power ** 4)
        delay_doppler_map = const * np.sqrt(gl) * gamma_0 * f

        delay_doppler_map_masked = ddm_mask_ranges(
            delay_doppler_map,
            self.ddm_masking,
            geo,
            p["Lx"],
            self.scenario.sar.span,
            rp.dr,
            beam_index
        )

        # compute the return power model
        waveform_power = bn.nansum(delay_doppler_map, 1) / len(beam_index)
        peak_power = bn.nanmax(waveform_power)

        if self.cfg.norm_model_power:
            waveform_model = wfm.amplitude_scale * (waveform_power/peak_power + wfm.thermal_noise)
        else:
            waveform_model = waveform_power.copy()
        # waveform_model_scaled_power = amplitude_scale * (pr / np.nanmax(pr)) + self.normed_waveform.thermal_noise

        # Compile the output
        return WaveformModelOutput(
            tau,
            waveform_model,
            wfm.amplitude_scale,
            delay_doppler_map,
            delay_doppler_map_masked,
            wfm.epoch,
            wfm.significant_wave_height,
            wfm.mean_square_slope,
            gamma_0
        )

    def _precompute_static_parameters(self) -> None:
        # TODO: Check if more parameters can be computed with fixed alpha power value not depended on SWH

        geo = self.scenario.geo
        rp = self.scenario.rp
        lut = self.lut
        beam_index = self.scenario.sar.beam_index

        p = {}

        p["Lx"] = CONSTANTS.c0 * geo.altitude / (2. * geo.velocity * rp.frequency * rp.pulses_per_burst * rp.pri_sar)
        if self.weighted and self.mode == 2:
            p["Lx"] *= self.weight_factor

        # Ly: pulse-limited radius
        p["Ly"] = np.sqrt(CONSTANTS.c0 * geo.altitude / (geo.kappa * rp.bandwidth))

        # Lz: vertical resolution
        p["Lz"] = CONSTANTS.c0 / (2. * rp.bandwidth)
        factor = 8. * np.log(2.)
        p["alpha_x"] = factor / (geo.altitude ** 2. * rp.beam_width_along ** 2.)
        p["alpha_y"] = factor / (geo.altitude ** 2. * rp.beam_width_across ** 2.)
        p["Lg"] = geo.kappa / (2. * geo.altitude * p["alpha_y"])
        p["xl"] = p["Lx"] * beam_index
        p["ls"] = self.flag_slope * geo.orbit_slope * geo.altitude / (geo.kappa * p["Lx"])
        p["xp"] = +geo.altitude * geo.pitch
        p["yp"] = -geo.altitude * geo.roll
        p["csi_max_F0"] = np.max(lut.f0[:, 0])
        p["csi_min_F0"] = np.min(lut.f0[:, 0])
        p["csi_max_F1"] = np.max(lut.f1[:, 0])
        p["csi_min_F1"] = np.min(lut.f1[:, 0])

        self.static_parameters = p

    def get_fit_params(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.fit_params) if self.collect_fit_params else None
