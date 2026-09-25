# -*- coding: utf-8 -*-
"""

"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import numpy as np

from samosa_waveform_model.constants import CONSTANTS


@dataclass
class SensorParameters:
    """ Information of the Radar Altimeter and Processing Configuration (hard coded to CryoSat-2 SIRAL SAR) """
    platform: str
    sensor: str
    mode: str
    pulses_per_burst: int  # number of pulses per burst [Np_burst]
    range_gates_per_pulse: int  # number of the range gates per pulse (without zero-padding) [Npulse]
    zero_padding_factor: int
    pulse_repetition_frequency: float  # Pulse Repetition Frequency in Hz (SAR mode) [PRF_SAR]
    burst_repetition_interval: float  # Burst Repetition Interval in sec [BRI]
    frequency: float  # Carrier Frequency in Hz [f_0]
    bandwidth: float  # Sampled Bandwidth in Hz [Bs]
    beam_width_along: float  # (rad) Antenna 3 dB beamwidth (along-track) [theta_3x]
    beam_width_across: float  # (rad) Antenna 3 dB beamwidth (cross-track) [theta_3y]
    alpha_power_ddm: float  # The alpha power for the delay doppler map (DDM) scaling
    alpha_power_ptr: float  # The alpha power for the PTR constant scaling
    num_look_min: float = -90.0
    num_look_max: float = 90.0
    # The default value of 1.0 leads to a fairly small number of doppler cells in the delay-doppler map (DDM).
    # If DDM mask is activated (by default), then the waveform may become choppy.
    beamsamp_factor: float = 2.0  # Number of beams per doppler cell


    @cached_property
    def pri_sar(self) -> float:
        return 1. / self.pulse_repetition_frequency

    @cached_property
    def wavelength(self) -> float:
        return CONSTANTS.c0 / self.frequency

    @cached_property
    def dfa(self) -> float:
        return self.pulse_repetition_frequency / self.pulses_per_burst

    @cached_property
    def dr(self) -> float:
        return CONSTANTS.c0 / (2. * self.bandwidth * self.zero_padding_factor)

    @cached_property
    def tau(self) -> np.ndarray:
        num_gates = self.range_gates_per_pulse * self.zero_padding_factor
        dt = 1. / (self.bandwidth * self.zero_padding_factor)
        return np.arange(-(num_gates / 2.) * dt, ((num_gates - 1) / 2) * dt, dt)


@dataclass
class SARParameters:

    beamsamp_factor: int = 1
    hamming_weighting: bool = True
    hamming_ptr_main_lobe_widening_factor: float = 1.4705
    look_angles: Optional[np.ndarray] = None
    doppler_frequencies: Optional[np.ndarray] = None
    span: Optional[np.ndarray] = None
    beam_index: Optional[np.ndarray] = None
    mask_ranges: Optional[np.ndarray] = None

    def compute_multi_look_parameters(
            self,
            geo: "PlatformLocation",
            sp: "SensorParameters"
    ) -> None:
        """
        Compute a set of beam parameters. Note the methods called depend on
        the order in which these are called

        :param geo: The platform location (including altitude and kappa factor)
        :param sp: The sensor parameters (including wavelength and burst repetition interval)

        :return: None: Changes parameters in place
        """

        # Compute default parameters if look angles has not been set to actual data
        if self.look_angles is None:
            self._compute_look_angles(
                geo.velocity,
                geo.altitude,
                geo.kappa,
                sp.burst_repetition_interval,
                sp.num_look_min,
                sp.num_look_max
            )
        self._compute_doppler_frequencies(geo.velocity, sp.wavelength)
        self._compute_beam_index(sp.beamsamp_factor, sp.dfa)

    def _compute_look_angles(
            self,
            velocity: float,
            altitude: float,
            kappa: float,
            burst_repetition_interval: float,
            num_look_min: float = -90.0,
            num_look_max: float = 90.0
    ) -> None:
        """
        Computes the look angles for a given height and velocity

        :param velocity:
        :param altitude:
        :param kappa:
        :param burst_repetition_interval:
        :param num_look_min
        :param num_look_max:

        :return:
        """
        dtheta = velocity * burst_repetition_interval / (altitude * kappa)
        theta1 = np.pi / 2. + dtheta * num_look_min
        theta2 = np.pi / 2. + dtheta * num_look_max
        self.look_angles = np.rad2deg(np.arange(theta1, theta2, dtheta))

    def _compute_doppler_frequencies(
            self,
            velocity: float,
            wavelength: float
    ) -> None:
        """
        Compute doppler frequencies

        :param velocity:
        :param wavelength:

        :return:
        """
        self.doppler_frequencies = (2 * velocity / wavelength) * np.cos(np.deg2rad(self.look_angles))

    def _compute_beam_index(
            self,
            beamsamp_factor: float,
            dfa: float
    ) -> None:
        beam_index = np.around(beamsamp_factor * self.doppler_frequencies / dfa) / beamsamp_factor
        print(f"beam_index: {beam_index.size}")
        self.span = np.where(np.diff(beam_index, axis=0) == 0)
        self.beam_index = np.delete(beam_index, self.span)
