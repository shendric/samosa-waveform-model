# -*- coding: utf-8 -*-

"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from warnings import warn
import bottleneck as bn
import pandas as pd
import numpy as np

from pydantic import BaseModel
from typing import Dict, Optional, Tuple, Union, Literal

from samosa_waveform_model.enums import WaveformModelEngines
from samosa_waveform_model.dataclasses import (SensorParameters, PlatformLocation, SARParameters,
                                               CONSTANTS, WaveformModelOutput, WaveformModelParameters)
from samosa_waveform_model.lut import SAMOSA_MODEL_TERMS_LUT, ALPHA_POWER_PTR_LUTS


from samosa_waveform_model.funcs_py import (compute_gl, compute_gamma0, compute_t_kappa, ddm_mask_ranges)


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
        Bundle of all input parameters for the SAMOSA+ waveform model.

        :param rp: Sensor (Radar) parameters
        :param geo: Location and attitude of the platform (satellite)
        :param sar: SAR processing parameters
        """
        self.rp = rp
        self.geo = geo
        self.sar = sar

    def compute_static_parameters(self, engine: WaveformModelEngines, flag_slope: int = 0) -> "FixedScenarioParameters":
        """
        Compute the static parameters that are independent of the waveform model parameters (SWH, MSS, epoch)
        and solely depends on the scenario data (sensor parameters, platform location, SAR parameters) and
        the waveform model configuration. Therefore, the computation of the static parameters is
        delayed and not done in the constructor of the ScenarioData class.

        :param engine: Either "samosa" or "samosa+"
        :param flag_slope: Integer flag to indicate whether to use the slope of the surface in the computation (0 for no slope, 1 for slope)

        :return: FixedScenarioParameters
        """
        return FixedScenarioParameters(engine, self, flag_slope)

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
            engine: Literal[WaveformModelEngines.SAMOSA, WaveformModelEngines.SAMOSAPLUS],
            swh: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Get the alpha power values for the delay doppler map scaling and the range PTR estimation.

        Notes to the use of `alpha_p` (`alpha_power_ptr`) and `alpha_power` (`alpha_power_ddm`)
        in SAMPy and this project:

        In the formulas in Dinardo 2020, there is just on alpha_p. Here, in the code
        the alpha_p goes into the computation of gl and alpha_power goes into
        the scaling factor for delay doppler map. The parameter `alpha_p` goes into compute_gl and
        `alpha_power` goes into the constant factor for the delay doppler map.

        According to a code comment in pysamosa (https://pypi.org/project/pysamosa/),
        `alpha_power` is an average and constant values which should be used for the
        delay doppler map scaling factor (thus the alpha_power lookup table in SAMPy
        only includes a singular alpha power value and is the same for Hamming and no Hamming).
        -> Renamed here to `alpha_power_ddm` and lookup table is no longer used in this project.

        `alpha_p`instead may vary as function of significant waveheight.
        But when zero-padding is applied, then alpha_p may also be constant
        value (section 3.2.3 in Dinardo et al. 2020, with a fixed value of 0.55).
        Nevertheless, SAMPy sets `alpha_p` to 0.42349 for SAMOSA+.
        But since the SAMOSA+ retracker uses SAMOSA (with nu set to zero) for the
        significant waveheight step, a lookup table for `alpha_p` is required
        for both Hamming and no-Hamming configurations.
        -> Renamed here to `alpha_power_ptr` (for the range PTR)

        :param engine: The waveform model engine to use (SAMOSA or SAMOSA+)
        :param swh: The significant wave height (SWH) required for the alpha power lookup table. Not needed
            for the SAMOSA+ engine, where a fixed alpha power value is used from the sensor configuration
            :parameter

        :raises ValueError: If the engine is not recognized or if the SWH is not provided for the SAMOSA engine.

        :return: A tuple of (alpha_power_ptr, alpha_power_ddm).
        """

        # If Hamming weighting is used, the DDM alpha power values is
        # the same for SAMOSA and SAMOSA+
        alpha_power_ddm = self.rp.alpha_power_ddm

        if self.sar.hamming_weighting:

            match engine:

                # [SAMOSA & hamming]: Get alpha power values for the range PTR estimation
                # from the lookup table for the SAMOSA engine
                case WaveformModelEngines.SAMOSA:
                    alpha_power_ptr = ALPHA_POWER_PTR_LUTS.get(self.rp.platform, swh, hamming=True)

                # [SAMOSA+ & hamming]: Get alpha power values for the range PTR estimation
                # from the sensor configuration
                case WaveformModelEngines.SAMOSAPLUS:
                    alpha_power_ptr = self.rp.alpha_power_ptr

                case _:
                    raise ValueError(f"Unknown waveform model engine: {engine}")

        # No Hamming
        else:
            alpha_power_ptr = ALPHA_POWER_PTR_LUTS.get(self.rp.platform, swh, hamming=False)

        return alpha_power_ptr, alpha_power_ddm


class FixedScenarioParameters(object):

    def __init__(
                self,
                engine: WaveformModelEngines,
                scenario: ScenarioData,
                flag_slope: int = 0,
    ) -> None:
            """
            A class for the pre-computation of fixed variables for the SAMOSA+ waveform model.

            This class exists for the purpose of fitting the waveform model to data,
            where the fixed variables can be pre-computed once and don't need to be computed
            in every iteration of the fitting process.

            :param engine: The waveform model engine to use (SAMOSA or SAMOSA+)
            :param scenario: The waveform model input data (sensor parameters, platform location, SAR parameters)
            :param flag_slope: Integer flag to indicate whether to use the orbit slope of the surface in the computation
                (0 for no slope, 1 for slope)
            """
            self.p = self.compute_static_parameters(engine, scenario, flag_slope)

    @staticmethod
    def compute_static_parameters(
            engine: WaveformModelEngines,
            scenario: ScenarioData,
            flag_slope: int
    ) -> Dict:
        """
        Pre-compute the parameters that are independent of the waveform model parameters (SWH, MSS, epoch)
        and solely depends on the scenario data (sensor parameters, platform location, SAR parameters)

        :param engine: The waveform model engine to use (SAMOSA or SAMOSA+)
        :param scenario: The waveform model input data (sensor parameters, platform location, SAR parameters)
        :param flag_slope: Whether to use the orbit slope in the computation (integer 0 for no slope, 1 for slope)

        :return: A dictionary of pre-computed static parameters
        """

        # short variable names for brevity
        geo = scenario.geo
        rp = scenario.rp
        sar = scenario.sar

        # Compute spatial and vertical resolution
        p = {"lx": CONSTANTS.c0 * geo.altitude / (2. * geo.velocity * rp.frequency * rp.pulses_per_burst * rp.pri_sar)}
        if sar.hamming_weighting and engine == WaveformModelEngines.SAMOSAPLUS:
            p["lx"] *= sar.hamming_ptr_main_lobe_widening_factor
        # Ly: pulse-limited radius
        p["ly"] = np.sqrt(CONSTANTS.c0 * geo.altitude / (geo.kappa * rp.bandwidth))
        # Lz: vertical resolution
        p["lz"] = CONSTANTS.c0 / (2. * rp.bandwidth)

        factor = 8. * np.log(2.)
        p["alpha_x"] = factor / (geo.altitude ** 2. * rp.beam_width_along ** 2.)
        p["alpha_y"] = factor / (geo.altitude ** 2. * rp.beam_width_across ** 2.)

        p["lg"] = geo.kappa / (2. * geo.altitude * p["alpha_y"])
        p["xl"] = p["lx"] * sar.beam_index
        p["ls"] = flag_slope * geo.orbit_slope * geo.altitude / (geo.kappa * p["lx"])
        p["xp"] = +geo.altitude * geo.pitch
        p["yp"] = -geo.altitude * geo.roll
        return p

    def __getitem__(self, item):
        """
        Provide dictionary-like access to the pre-computed static parameters.

        :param item: Item name

        :raises KeyError: If the item is not found in the pre-computed parameters.

        :return: Value
        """
        if item in self.p:
            return self.p[item]
        else:
            raise KeyError(f"Item '{item}' not found in pre-computed static parameters [{self.p.keys()}].")


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


class SAMOSAWaveformModelConfig(BaseModel):
    """
    A class for the configuration of the SAMOSA+ waveform model.
    """
    use_slope: bool = False
    norm_model_power: bool = True
    use_masked_ddm: bool = True

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
            engine: Union[str, WaveformModelEngines],
            scenario: ScenarioData,
            use_slope: bool = False,
            norm_model_power: bool = True,
            use_masked_ddm: bool = True,
            collect_fit_params: bool = False
    ) -> None:
        """
        Initialize the SAMOSA/SAMOSA+ waveform model with the given scenario data and configuration parameters.

        Static parameters that are independent of the waveform model parameters (SWH, MSS, epoch) and solely depend
        on the scenario data (sensor parameters, platform location, SAR parameters) and the SAMOSA waveform model
        configuration keywords are pre-computed and stored in the `self.scenario.static_parameters`.

        To compute the actual waveform model, call the `generate_delay_doppler_waveform` method with
        the desired waveform model parameters.

        :param engine: Either `samosa` or `samosa+` to select the waveform model engine.
            - `samosa`: Uses the assumption of an infinite diffusive surface (nu = 1/mss = 0)
               and selects different alpha power values for the delay doppler map scaling and the range
               PTR estimation. In the original SAMPy implementation, this waveform model
               is used for the significant waveheight estimation.
            - `samosa+`: Allows finite mss values (nu = 1/mss > 0) and uses a fixed alpha power value for
               both the delay doppler map scaling and the range PTR estimation. In the original SAMPy implementation,
               this waveform model is used for the epoch and mean square slope estimation.
        :param scenario: The scenario data for the waveform model including sensor parameters, platform location,
            and SAR parameters.
        :param use_slope: Whether to use the slope of the surface in the computation (boolean).
        :param norm_model_power: Whether to normalize the model power of the computed waveform (boolean).
        :param use_masked_ddm: Whether to use a masked delay doppler map for waveform computation (boolean).
        :param collect_fit_params: Whether to collect fit parameters (boolean).
        """

        # Store the input parameters with basic sanity check
        if isinstance(engine, str):
            engine = WaveformModelEngines(engine)
        assert isinstance(engine, WaveformModelEngines), "engine must be an instance of WaveformModelEngines"
        self.engine = engine
        assert isinstance(scenario, ScenarioData), "scenario must be an instance of ScenarioData"
        self.scenario = scenario

        # Store the configuration parameters for the SAMOSA+ waveform model
        # (In separate class to reduce attribute clutter)
        self.cfg = SAMOSAWaveformModelConfig(
            use_slope=use_slope,
            norm_model_power=norm_model_power,
            use_masked_ddm=use_masked_ddm
        )

        # Pre-compute the parameters that are independent of the waveform model parameters (SWH, MSS, epoch)
        # and solely depends on the scenario data (sensor parameters, platform location, SAR parameters)
        # and the SAMOSA waveform model configuration.
        # NOTE: This is done for efficiency during repeated computations of the waveform model
        #       with the same scenario (e.g. during fitting)
        self.static_parameters = self.scenario.compute_static_parameters(engine, self.cfg.flag_slope)

        # Store the lookup tables for the SAMOSA+ waveform model
        # NOTE: The f0 and f1 lookup table files have been read and stored
        #       during the package initialization to avoid multiple reading of the files
        #       by repeated calls to the waveform model
        self.model_term_lut = SAMOSA_MODEL_TERMS_LUT

        # Housekeeping variables (For waveform fitting)
        # NOTE: These are only used to store fitting parameters
        #       used for each waveform model computation for each iteration
        #       and are not required for the waveform model itself
        self.fit_params = SAMOSAFitHouseKeeping(collect_fit_params)

    def generate_delay_doppler_waveform(
            self,
            waveform_model_parameters: "WaveformModelParameters",
    ) -> "WaveformModelOutput":
        """
        Compute a delay doppler waveform for the given waveform model parameters.

        NOTE: This method is derived from sampy.SAMOSA.__Generate_SamosaDDM in the
        original SAMPy implementation of the SAMOSA+ waveform model.

        :param waveform_model_parameters:

        :return: The computed waveform model output including the delay doppler map,
            the waveform power, and other parameters as a dataclass.
        """

        # Create short variables name for brevity of some parameters
        geo = self.scenario.geo
        rp = self.scenario.rp
        wfm = waveform_model_parameters
        swh = float(wfm.significant_wave_height)
        nu = float(wfm.nu)
        alt = float(geo.altitude)
        tau = self.scenario.rp.tau - wfm.epoch
        beam_index = self.scenario.sar.beam_index
        p = self.static_parameters

        # Some sanity checks on the input parameters
        if self.engine == WaveformModelEngines.SAMOSA:
            # For SAMOSA engine, nu must be set to 0 (infinite diffusive surface)
            try:
                assert nu < 1e-12, "SAMOSA: nu must be 0 (infinite diffusive surface)"
            except AssertionError as e:
                warn(f"{e} -> Forcing nu to 0.")
                nu = 0.0

        # --- Collect the waveform model parameters if requested ---
        # This is useful to restore the parameters variations
        # in an optimization process
        if self.fit_params.collect_fit_params:
            self.fit_params.append(wfm)

        dk = (tau * rp.bandwidth)
        yk = 0 * dk
        dk_positive = np.where(dk > 0)
        yk[dk_positive] = p["ly"] * np.sqrt(dk[dk_positive])

        sigma_s = (swh / (4. * p["lz"]))

        # surface elevation standard deviation
        sigma_z = (swh / 4.)

        # Get the alpha power values for the delay doppler map scaling and the range PTR estimation
        alpha_power_ptr, alpha_power_ddm = self.scenario.get_alpha_power(self.engine, swh)

        gl = compute_gl(alpha_power_ptr, p["lx"], p["ly"], p["lz"], beam_index, p["ls"], swh)

        xi = gl[None, :] * dk[:, None]
        z = 1. / 4. * xi ** 2

        # gamma0: Surface backscatter response
        gamma_0 = compute_gamma0(p["alpha_y"], p["yp"], p["alpha_x"], nu, alt, p["xl"], p["xp"], yk)

        # Equation 3.19 in Dinardo
        t_kappa = compute_t_kappa(z, dk, nu, alt, p["alpha_y"], p["yp"], p["ly"])

        # f0: zero order term of the SAMOSA SAR return waveform model
        f0 = self.model_term_lut.get(order=0, xi=xi, z=z)

        # f1: first order term of the SAMOSA SAR return waveform model
        f1 = self.model_term_lut.get(order=1, xi=xi, z=z)

        f = (f0 + sigma_z / p["lg"] * t_kappa * gl * sigma_s * f1)

        # Compute the delay doppler map (ddm)
        # TODO: Where is **4 in the const coming from? It is **2 in eq 3.15 in Dinardo 2020
        const = np.sqrt(2. * np.pi * alpha_power_ddm ** 4)
        delay_doppler_map = const * np.sqrt(gl) * gamma_0 * f

        # Limit the delay doppler map to the valid ranges defined by the SAR mask ranges
        # TODO: This function has been implemented for CryoSat-2, check for other missions
        delay_doppler_map_masked = ddm_mask_ranges(
            delay_doppler_map,
            self.scenario.sar.mask_ranges,
            geo,
            p["lx"],
            self.scenario.sar.span,
            rp.dr,
            beam_index
        )

        # compute the return power model
        match self.cfg.use_masked_ddm:
            case True:
                waveform_power = bn.nansum(delay_doppler_map_masked, 1) / len(beam_index)
            case False:
                waveform_power = bn.nansum(delay_doppler_map, 1) / len(beam_index)

        peak_power = bn.nanmax(waveform_power)
        if self.cfg.norm_model_power:
            waveform_model = wfm.amplitude_scale * (waveform_power/peak_power + wfm.thermal_noise)
        else:
            waveform_model = waveform_power.copy()

        # Compile the output
        return WaveformModelOutput(
            tau,
            waveform_model,
            peak_power,
            delay_doppler_map,
            delay_doppler_map_masked,
            wfm.epoch,
            wfm.significant_wave_height,
            wfm.mean_square_slope,
            wfm.amplitude_scale,
            gamma_0
        )

    def get_fit_params(self) -> Optional[pd.DataFrame]:
        return pd.DataFrame(self.fit_params.fit_params) if self.fit_params.collect_fit_params else None
