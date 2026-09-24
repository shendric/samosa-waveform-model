# -*- coding: utf-8 -*-

"""
"""

__author__ = "Stefan Hendricks <stefan.hendricks@awi.de>"

import numpy as np
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt

from samosa_waveform_model.dataclasses import CONSTANTS
from samosa_waveform_model.lut import __LUT_PATH__, SAMOSA_MODEL_TERMS_LUT


__THIS_DIR__ = Path(__file__).parent


def main():

    # Copied from cryosat2 sar example
    xi_example = read_csv(__THIS_DIR__ / "xi.csv", header=None).to_numpy()

    f0_old, f1_old = compute_f0_f1_old(xi_example)
    f0_new, f1_new = compute_f0_f1_new(xi_example)

    import matplotlib.pyplot as plt

    plt.figure("f0 diff", dpi=150, figsize=(5, 5))
    plt.imshow(f0_new-f0_old, label="f0")
    plt.colorbar()

    plt.figure("f1 diff", dpi=150, figsize=(5, 5))
    plt.imshow(f1_new-f1_old, label="f1")
    plt.colorbar()
    plt.show()


def compute_f0_f1_old(xi_example):

    kwargs = dict(dtype='float', comments='#', delimiter=None)
    f0_lut = np.genfromtxt(__LUT_PATH__ / "LUT_F0.txt", **kwargs)
    f1_lut = np.genfromtxt(__LUT_PATH__ / "LUT_F1.txt", **kwargs)

    csi_min_f0 = np.min(f0_lut[:, 0])
    csi_max_f0 = np.max(f0_lut[:, 0])
    csi_min_f1 = np.min(f1_lut[:, 0])
    csi_max_f1 = np.max(f1_lut[:, 0])
    z = 1. / 4. * xi_example ** 2
    f0 = compute_f0(xi_example, csi_min_f0, csi_max_f0, z, f0_lut)
    f1 = compute_f1(xi_example, csi_min_f1, csi_max_f1, z, f1_lut)
    return f0, f1


def compute_f0_f1_new(xi_example):
    z = 1. / 4. * xi_example ** 2
    f0 = SAMOSA_MODEL_TERMS_LUT.get(0, xi_example, z)
    f1 = SAMOSA_MODEL_TERMS_LUT.get(1, xi_example, z)
    return f0, f1


def compute_f0(csi, csi_min_f0, csi_max_f0, z, lut):
    f0 = get_clipped_f0(csi, csi_min_f0, csi_max_f0, lut)
    f0 = set_f0_z_idx_max_f0(csi, csi_max_f0, f0, z)
    f0 = set_f0_csi_eq_0(f0, csi)
    f0 = set_f0_csi_lt_csi_min(f0, csi, csi_min_f0)
    return f0


def compute_f1(csi, csi_min_f1, csi_max_f1, z, lut):
    f1 = get_clipped_f1(csi, csi_min_f1, csi_max_f1, lut)
    f1 = set_f1_csi_gt_csi_max(f1, csi, csi_max_f1, z)
    f1 = set_f1_csi_eq_0(f1, csi)
    f1 = set_f1_csi_lt_csi_min(f1, csi, csi_min_f1)
    return f1


def get_clipped_f0(csi, csi_min_fi, csi_max_fi, lut_fi):
    fi = np.zeros(np.shape(csi))
    clip_fi = np.bitwise_and(csi >= csi_min_fi, csi <= csi_max_fi)
    idx = np.floor((lut_fi[:, 0].size - 1) * ((csi[clip_fi] - csi_min_fi) / (csi_max_fi - csi_min_fi))).astype(int)
    fi[clip_fi] = (csi[clip_fi] - lut_fi[idx, 0]) * ((lut_fi[idx + 1, 1] - lut_fi[idx, 1]) / (
            lut_fi[idx + 1, 0] - lut_fi[idx, 0])) + lut_fi[idx, 1]
    return fi


def get_clipped_f1(csi, csi_min_fi, csi_max_fi, lut_fi):
    fi = np.zeros(np.shape(csi))
    clip_fi = np.bitwise_and(csi >= csi_min_fi, csi <= csi_max_fi)
    idx = np.floor((lut_fi[:, 0].size - 1) * ((csi[clip_fi] - csi_min_fi) / (csi_max_fi - csi_min_fi))).astype(int)
    fi[clip_fi] = (csi[clip_fi] - lut_fi[idx, 0]) * ((lut_fi[idx + 1, 1] - lut_fi[idx, 1]) / (
            lut_fi[idx + 1, 0] - lut_fi[idx, 0])) + lut_fi[idx, 1]
    return fi


def set_f0_z_idx_max_f0(csi, csi_max_f0, f0, z):
    idx_max_f0 = find_idx_max_f0(csi, csi_max_f0)
    z_idx_max_f0 = z[idx_max_f0]
    f0[idx_max_f0] = 1. / 2. * CONSTANTS.sqrt_pi / z_idx_max_f0 ** (1. / 4.) * (
            1. + 3. / (32. * z_idx_max_f0) + 105. / (
            2048. * z_idx_max_f0 ** 2) + 10395. / (
                    196608. * z_idx_max_f0 ** 3))
    return f0


def find_idx_max_f0(csi, csi_max_f0):
    return np.where(csi > csi_max_f0)


def set_f0_csi_eq_0(f0, csi):
    f0[np.where(csi == 0)] = CONSTANTS.f0_csi_0
    return f0


def set_f1_csi_eq_0(f1, csi):
    f1[np.where(csi == 0)] = CONSTANTS.f1_csi_0
    return f1


def set_f0_csi_lt_csi_min(f0, csi, csi_min_f0):
    f0[np.where(csi < csi_min_f0)] = 0
    return f0


def set_f1_csi_lt_csi_min(f1, csi, csi_min_f1):
    f1[np.where(csi < csi_min_f1)] = 0.0
    return f1


def set_f1_csi_gt_csi_max(f1, csi, csi_max_f1, z):
    idx_max_f1 = np.where(csi > csi_max_f1)
    f1[idx_max_f1] = (1. / 2.) * 1. / 4. * CONSTANTS.sqrt_pi / (z[idx_max_f1]) ** (3. / 4.)
    return f1




if __name__ == "__main__":
    main()
