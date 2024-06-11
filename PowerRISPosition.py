# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:35:14 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Script to move the RIS around with fixed SP cluster position,
           and compute the SNR for each RIS position. (21/12/2023)
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import toml

from system import channel
if os.path.abspath("..")+"/Modules" not in sys.path:
    sys.path.append(os.path.abspath("..")+"/Modules")
from utilities import make_3D_grid


def main():
    # =============================================================================
    # Load configuration file
    # =============================================================================
    config_file = "system_config.toml"
    toml_in = toml.load(config_file)
    toml_settings = toml_in["settings"]
    toml_positions = toml_in["positions"]
    toml_estimation = toml_in["estimation"]
    chn = channel(None, False, **toml_settings)

    # =============================================================================
    # Implement setting dictionaries
    # =============================================================================
    np.random.seed(toml_settings["seed"])

    Phi_taus = np.array(toml_positions["Phi_taus"])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array(toml_positions["Phi_azs"])
    Phi_els = np.array(toml_positions["Phi_els"])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    print("Targets: \n", Phi)

    # =============================================================================
    # Make prior
    # =============================================================================
    sU = np.array(toml_settings["sU"])
    pU, oU = sU[:3], sU[3:]
    center = np.mean(Phi, axis=0)
    res_scale = toml_estimation["res_scale"]
    ResRegion = chn.sensor_resolution_function(center, sU, res_scale)

    theta0, thetal = chn.nonRIS_angles(np.expand_dims(center, axis=0), sU)
    _, theta_ResRegion = chn.nonRIS_angles(ResRegion, sU)
    tau_ResRegion, _ = chn.nonRIS_delays(ResRegion, pU)
    _, tau_bar_ResRegion, _ = chn.RIS_delays(ResRegion, pU)
    theta_bounds = (np.min(theta_ResRegion, axis=0)[0], np.max(theta_ResRegion, axis=0)[0],
                    np.min(theta_ResRegion, axis=0)[1], np.max(theta_ResRegion, axis=0)[1])
    tau_bounds = (np.min(tau_ResRegion), np.max(tau_ResRegion))
    tau_bar_bounds = (np.min(tau_bar_ResRegion), np.max(tau_bar_ResRegion))

    window = [0, 25, 0, 30, 1.6, 1.7]
    grid_dim = [30, 30, 1]
    pRflat, x_points, y_points, z_points = make_3D_grid(window, grid_dim)
    # sRflat = np.array([[44.0, 20.0, 29.0, 0.0, 0.0, 0.0]])

    SNRR = np.zeros(pRflat.shape[0])
    phiA = np.zeros(pRflat.shape[0])
    for idx, pR in enumerate(pRflat):
        sims = 1
        for i in range(sims):
            sR = np.hstack((pR, np.zeros(3)))
            toml_settings["sR"] = sR
            chn = channel(None, False, **toml_settings)

            phi_bounds = chn.RIS_angle_resolution_region(pU, ResRegion)
            phi0, _ = chn.RIS_angles(np.expand_dims(center, axis=0), sU[:3])
            prior = {"thetal": thetal[0], "phi0": phi0, "theta0": theta0,
                     "tau_bounds": tau_bounds, "theta_bounds": theta_bounds,
                     "tau_bar_bounds": tau_bar_bounds, "phi_bounds": phi_bounds}

            yN, yR, WN, WR, omega, f = chn.main_signal_model(np.expand_dims(center, axis=0), sU, prior)

            SNRR[idx] += np.linalg.norm(yR)**2/np.prod(yR.shape) * 2/chn.p_noise
            phiA[idx] = (phi_bounds[1] - phi_bounds[0]) * (phi_bounds[3] - phi_bounds[2])
        SNRR[idx] = 10*np.log10(SNRR[idx]/sims) - 30
    SNRR = SNRR.reshape((grid_dim[0], grid_dim[1]))
    phiA = phiA.reshape((grid_dim[0], grid_dim[1]))

    spec = plt.pcolormesh(x_points, y_points, phiA.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="Area [rad^2]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNRR.T, cmap="cool", shading="gouraud")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("")
    plt.show()


if __name__ == "__main__":
    main()

