import bottleneck as bn
import pandas as pd
import numpy as np
cimport numpy as np

import cython
cimport cython

from samosa_waveform_model.dataclasses import (
    CONSTANTS,
    WaveformModelOutput,
)
from samosa_waveform_model.lut import CS2_LOOKUP_TABLES

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.npy_bool DTYPE_BOOL_t
ctypedef np.int32_t DTYPE_LONG_t


cdef class SAMOSAWaveformModel:
    """
    Cython implementation of the SAMOSA+ waveform model.
    The original Python implementation remains in `model.py`.
    """
    cdef object scenario
    cdef object engine
    cdef int flag_slope
    cdef bint weighted
    cdef double weight_factor
    cdef int mode
    cdef object mask_ranges
    cdef object lut
    cdef object static_parameters
    cdef bint collect_fit_params
    cdef object fit_params
    cdef int generate_ddm_counter

    def __init__(
        self,
        scenario,
        engine="samosa+",
        use_slope=False,
        weighted=False,
        weight_factor=1.4705,
        mask_ranges=None,
        mode=1,
        collect_fit_params=False,
    ):
        self.scenario = scenario
        self.engine = engine
        self.flag_slope = int(use_slope)
        self.weighted = weighted
        self.weight_factor = weight_factor
        self.mode = mode
        self.mask_ranges = mask_ranges
        self.lut = CS2_LOOKUP_TABLES
        self.static_parameters = {}
        self.collect_fit_params = collect_fit_params
        self.fit_params = []
        self.generate_ddm_counter = 0
        self.set_mode(self.mode)

    def set_mode(self, mode_num):
        if mode_num not in [1, 2]:
            raise ValueError(f"mode number {mode_num} not in [1, 2")
        self.mode = mode_num
        self._precompute_static_parameters()

    def get_alpha_power(self, swh):
        if self.weighted:
            if self.mode == 1:
                ind = bn.nanargmin(abs(self.lut.alphap_weight[:, 0] - swh))
                alpha_p = self.lut.alphap_weight[:, 1][ind]
                alpha_power = 0.47356
            elif self.mode == 2:
                alpha_p = 0.42349
                alpha_power = 0.47356
            else:
                raise ValueError(f"Invalid mode: {self.mode} (must be 1 or 2)")
        else:
            alpha_p, alpha_power = self.get_alpha_power_no_weights(swh)
        return alpha_p, alpha_power

    def get_alpha_power_no_weights(self, swh):
        ind = np.argmin(abs(self.lut.alphap_noweight[:, 0] - swh))
        alpha_p = self.lut.alphapower_noweight[:, 1][ind]

        ind = np.argmin(abs(self.lut.alphapower_noweight[:, 0] - swh))
        alpha_power = self.lut.alphapower_noweight[:, 1][ind]
        return alpha_p, alpha_power

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] _compute_gamma0(
        self,
        double alpha_y,
        double yp,
        double alpha_x,
        double nu,
        double alt,
        np.ndarray[DTYPE_t, ndim=1] xl,
        double xp,
        np.ndarray[DTYPE_t, ndim=1] yk,
    ):
        cdef np.ndarray[DTYPE_t, ndim=2] xl_ = xl[None, :]
        cdef np.ndarray[DTYPE_t, ndim=2] yk_ = yk[:, None]
        cdef double alt2 = alt ** 2
        return np.exp(
            -alpha_y * yp ** 2
            - alpha_x * (xl_ - xp) ** 2
            - xl_ ** 2 * nu / alt2
            - (alpha_y + nu / alt2) * yk_ ** 2
        ) * np.cosh(2.0 * alpha_y * yp * yk_)

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] _compute_t_kappa(
        self,
        np.ndarray[DTYPE_t, ndim=2] z,
        np.ndarray[DTYPE_t, ndim=1] dk,
        double nu,
        double alt,
        double alpha_y,
        double yp,
        double ly,
    ):
        cdef np.ndarray[DTYPE_t, ndim=2] t_kappa = np.zeros(np.shape(z), dtype=np.float64)
        cdef np.ndarray[DTYPE_BOOL_t, ndim=1] dk_positive = dk > 0
        dk_positive_idx = np.where(dk_positive)
        dk_negative_idx = np.where(np.logical_not(dk_positive))
        dk_positive_sqrt = np.sqrt(dk[dk_positive_idx])

        t_kappa[dk_positive_idx, :] = (
            (1.0 + nu / ((alt ** 2) * alpha_y))
            - yp / (ly * dk_positive_sqrt) * np.tanh(2.0 * alpha_y * yp * ly * dk_positive_sqrt)[None, :]
        ).T
        t_kappa[dk_negative_idx, :] = (1.0 + nu / ((alt ** 2) * alpha_y)) - 2.0 * alpha_y * yp ** 2
        return t_kappa

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] _compute_f0(
        self,
        np.ndarray[DTYPE_t, ndim=2] csi,
        double csi_min_f0,
        double csi_max_f0,
        np.ndarray[DTYPE_t, ndim=2] z,
        lut,
    ):
        cdef np.ndarray[DTYPE_t, ndim=2] f0 = np.zeros(np.shape(z), dtype=np.float64)
        cdef np.ndarray[DTYPE_BOOL_t, ndim=2] clip_f0 = np.bitwise_and(csi >= csi_min_f0, csi <= csi_max_f0)
        cdef np.ndarray[DTYPE_LONG_t, ndim=1] idx = np.floor(
            (len(lut.f0[:, 0]) - 1) * ((csi[clip_f0] - csi_min_f0) / (csi_max_f0 - csi_min_f0))
        ).astype(np.int32)
        f0[clip_f0] = (csi[clip_f0] - lut.f0[idx, 0]) * (
            (lut.f0[idx + 1, 1] - lut.f0[idx, 1]) / (lut.f0[idx + 1, 0] - lut.f0[idx, 0])
        ) + lut.f0[idx, 1]

        idx_max_f0 = np.where(csi > csi_max_f0)
        z_max_f0 = z[idx_max_f0]
        f0[idx_max_f0] = 0.5 * np.sqrt(np.pi) / (z_max_f0) ** (1.0 / 4.0) * (
            1.0
            + 3.0 / (32.0 * z_max_f0)
            + 105.0 / (2048.0 * (z[(csi > csi_max_f0)]) ** 2)
            + 10395.0 / (196608.0 * (z_max_f0) ** 3)
        )
        f0[np.where(csi == 0)] = CONSTANTS.f0_csi_0
        f0[np.where(csi < csi_min_f0)] = 0.0
        return f0

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] _compute_f1(
        self,
        np.ndarray[DTYPE_t, ndim=2] csi,
        double csi_min_f1,
        double csi_max_f1,
        np.ndarray[DTYPE_t, ndim=2] z,
        lut,
    ):
        cdef np.ndarray[DTYPE_BOOL_t, ndim=2] clip_f1 = np.bitwise_and(csi >= csi_min_f1, csi <= csi_max_f1)
        cdef np.ndarray[DTYPE_t, ndim=2] f1 = np.zeros(np.shape(z), dtype=np.float64)
        cdef np.ndarray[DTYPE_LONG_t, ndim=1] idx = np.floor(
            (len(lut.f1[:, 0]) - 1) * ((csi[clip_f1] - csi_min_f1) / (csi_max_f1 - csi_min_f1))
        ).astype(np.int32)
        f1[clip_f1] = (csi[clip_f1] - lut.f1[idx, 0]) * (
            (lut.f1[idx + 1, 1] - lut.f1[idx, 1]) / (lut.f1[idx + 1, 0] - lut.f1[idx, 0])
        ) + lut.f1[idx, 1]
        idx_max_f1 = np.where(csi > csi_max_f1)
        f1[idx_max_f1] = 0.5 * 0.25 * np.sqrt(np.pi) / (z[idx_max_f1]) ** (3.0 / 4.0)
        f1[np.where(csi == 0)] = CONSTANTS.f1_csi_0
        f1[np.where(csi < csi_min_f1)] = 0.0
        return f1

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=1] _compute_gl(
        self,
        double alpha_p,
        double lx,
        double ly,
        double lz,
        np.ndarray[DTYPE_t, ndim=1] l,
        double ls,
        double swh,
    ):
        return 1.0 / np.sqrt(
            alpha_p ** 2 + 4.0 * (alpha_p ** 2) * (lx / ly) ** 4 * (l - ls) ** 2 + np.sign(swh) * (swh / (4.0 * lz)) ** 2
        )

    @cython.boundscheck(False)
    @cython.wraparound(True)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_t, ndim=2] _ddm_mask_ranges(
        self,
        np.ndarray[DTYPE_t, ndim=2] ddm,
        mask_ranges,
        geo,
        double lx,
        span,
        double dr,
        np.ndarray[DTYPE_t, ndim=1] beam_index,
    ):
        if mask_ranges is None:
            mask_ranges_demin = geo.altitude * (np.sqrt(1 + (geo.kappa * ((lx * beam_index) / geo.altitude) ** 2)) - 1)
        else:
            mask_ranges = np.delete(mask_ranges, span)
            mask_ranges_demin = mask_ranges - min(mask_ranges)
        cdef int num_range_gates = ddm.shape[0]
        cdef np.ndarray[DTYPE_t, ndim=2] r = np.tile(mask_ranges_demin, (num_range_gates, 1))
        cdef np.ndarray[DTYPE_t, ndim=2] dr_tiled = np.tile(
            dr * np.arange(num_range_gates - 1, -1, -1), (len(beam_index), 1)
        ).T
        cdef np.ndarray[DTYPE_t, ndim=2] ddm_masked = ddm.copy()
        ddm_masked[np.where(r >= dr_tiled)] = 0
        return ddm_masked

    def generate_delay_doppler_waveform(self, waveform_model_parameters, norm_model_power=True):
        geo = self.scenario.geo
        rp = self.scenario.rp
        wfm = waveform_model_parameters
        lut = self.lut
        swh = wfm.significant_wave_height
        nu = wfm.nu
        alt = geo.altitude
        tau = self.scenario.rp.tau - wfm.epoch
        beam_index = self.scenario.sar.beam_index

        if self.collect_fit_params:
            self.fit_params.append(wfm)

        p = self.static_parameters

        dk = (tau * rp.bandwidth)
        yk = 0 * dk
        dk_positive = np.where(dk > 0)
        yk[dk_positive] = p["Ly"] * np.sqrt(dk[dk_positive])

        sigma_s = (swh / (4.0 * p["Lz"]))
        sigma_z = (swh / 4.0)

        alpha_p, alpha_power = self.get_alpha_power(swh)
        gl = self._compute_gl(alpha_p, p["Lx"], p["Ly"], p["Lz"], beam_index, p["ls"], swh)
        csi = gl[None, :] * dk[:, None]
        z = 0.25 * csi ** 2

        gamma_0 = self._compute_gamma0(p["alpha_y"], p["yp"], p["alpha_x"], nu, alt, p["xl"], p["xp"], yk)
        t_kappa = self._compute_t_kappa(z, dk, nu, alt, p["alpha_y"], p["yp"], p["Ly"])
        f0 = self._compute_f0(csi, p["csi_min_F0"], p["csi_max_F0"], z, lut)
        f1 = self._compute_f1(csi, p["csi_min_F1"], p["csi_max_F1"], z, lut)
        f = (f0 + sigma_z / p["Lg"] * t_kappa * gl * sigma_s * f1)

        const = np.sqrt(2.0 * np.pi * alpha_power ** 4)
        delay_doppler_map = const * np.sqrt(gl) * gamma_0 * f

        delay_doppler_map_masked = self._ddm_mask_ranges(
            delay_doppler_map,
            self.mask_ranges,
            geo,
            p["Lx"],
            self.scenario.sar.span,
            rp.dr,
            beam_index,
        )

        waveform_power = bn.nansum(delay_doppler_map, 1) / len(beam_index)
        peak_power = bn.nanmax(waveform_power)

        if norm_model_power:
            waveform_model = wfm.amplitude_scale * (waveform_power / peak_power + wfm.thermal_noise)
        else:
            waveform_model = waveform_power.copy()

        self.generate_ddm_counter += 1

        return WaveformModelOutput(
            tau,
            waveform_model,
            wfm.amplitude_scale,
            delay_doppler_map,
            delay_doppler_map_masked,
            wfm.epoch,
            wfm.significant_wave_height,
            wfm.mean_square_slope,
            gamma_0,
        )

    def _precompute_static_parameters(self):
        geo = self.scenario.geo
        rp = self.scenario.rp
        lut = self.lut
        beam_index = self.scenario.sar.beam_index

        p = {}
        p["Lx"] = CONSTANTS.c0 * geo.altitude / (2.0 * geo.velocity * rp.frequency * rp.pulses_per_burst * rp.pri_sar)
        if self.weighted and self.mode == 2:
            p["Lx"] *= self.weight_factor

        p["Ly"] = np.sqrt(CONSTANTS.c0 * geo.altitude / (geo.kappa * rp.bandwidth))
        p["Lz"] = CONSTANTS.c0 / (2.0 * rp.bandwidth)
        factor = 8.0 * np.log(2.0)
        p["alpha_x"] = factor / (geo.altitude ** 2.0 * rp.beam_width_along ** 2.0)
        p["alpha_y"] = factor / (geo.altitude ** 2.0 * rp.beam_width_across ** 2.0)
        p["Lg"] = geo.kappa / (2.0 * geo.altitude * p["alpha_y"])
        p["xl"] = p["Lx"] * beam_index
        p["ls"] = self.flag_slope * geo.orbit_slope * geo.altitude / (geo.kappa * p["Lx"])
        p["xp"] = +geo.altitude * geo.pitch
        p["yp"] = -geo.altitude * geo.roll
        p["csi_max_F0"] = np.max(lut.f0[:, 0])
        p["csi_min_F0"] = np.min(lut.f0[:, 0])
        p["csi_max_F1"] = np.max(lut.f1[:, 0])
        p["csi_min_F1"] = np.min(lut.f1[:, 0])

        self.static_parameters = p

    def get_fit_params(self):
        return pd.DataFrame(self.fit_params) if self.collect_fit_params else None
