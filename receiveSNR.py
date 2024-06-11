# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:52:53 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Functionality: individual_power() and main_power().
           The individual_power() function computes SNR of each type of path.
           main_power() computes SNR for the main signal model. (11/12/2023)
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


def main_power():
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
    sU = np.array(toml_settings["sU"])
    sR = np.array(chn.sR)
    lambda_ = chn.lambda_
    NR = chn.NR

    Phi_taus = np.array(toml_positions["Phi_taus"])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array(toml_positions["Phi_azs"])
    Phi_els = np.array(toml_positions["Phi_els"])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    print("Targets: \n", Phi)
    triu_idx = np.triu_indices(Phi.shape[0], k=1)
    dists = np.linalg.norm(Phi[None, :, :] - Phi[:, None, :], axis=2)
    print("Distances between targets: \n", dists[triu_idx])
    dist_RIS = np.linalg.norm(sR[None, :3] - Phi, axis=1)
    print("Distance to RIS: \n", dist_RIS)
    Fraunhofer_dist = 2*(NR[0]*lambda_/4)**2/lambda_
    print("Fraunhofer distance: \n", Fraunhofer_dist)

    # =============================================================================
    # Make prior
    # =============================================================================
    pU, oU = sU[:3], sU[3:]
    center = np.mean(Phi, axis=0)
    rcs = None
    res_scale = toml_estimation["res_scale"]
    ResRegion0 = chn.sensor_resolution_function(center, sU, 2*res_scale)
    ResRegion = chn.sensor_resolution_function(center, sU, res_scale)
    phi_bounds = chn.RIS_angle_resolution_region(pU, ResRegion0)
    phi0, _ = chn.RIS_angles(np.expand_dims(center, axis=0), sU[:3])
    theta0, thetal = chn.nonRIS_angles(np.expand_dims(center, axis=0), sU)
    _, theta_ResRegion = chn.nonRIS_angles(ResRegion, sU)
    tau_ResRegion, _ = chn.nonRIS_delays(ResRegion, pU)
    _, tau_bar_ResRegion, _ = chn.RIS_delays(ResRegion, pU)
    theta_bounds = (np.min(theta_ResRegion, axis=0)[0], np.max(theta_ResRegion, axis=0)[0],
                    np.min(theta_ResRegion, axis=0)[1], np.max(theta_ResRegion, axis=0)[1])
    tau_bounds = (np.min(tau_ResRegion), np.max(tau_ResRegion))
    tau_bar_bounds = (np.min(tau_bar_ResRegion), np.max(tau_bar_ResRegion))

    prior = {"thetal": thetal[0], "phi0": phi0, "theta0": theta0,
             "tau_bounds": tau_bounds, "theta_bounds": theta_bounds,
             "tau_bar_bounds": tau_bar_bounds, "phi_bounds": phi_bounds}

    # =============================================================================
    # Power on prior area
    # =============================================================================
    # window = [np.min(ResRegion0, axis=0)[0]-10, np.max(ResRegion0, axis=0)[0]+10,
    #           np.min(ResRegion0, axis=0)[1]-10, np.max(ResRegion0, axis=0)[1]+10,
    #           np.min(ResRegion0, axis=0)[2]-10, np.max(ResRegion0, axis=0)[2]+10]
    window = [0, 25, 0, 30, 12, 13]
    grid_dim = [30, 30, 1]
    grid_flat, x_points, y_points, z_points = make_3D_grid(window, grid_dim)
    C1, C2, C3 = np.meshgrid(x_points, y_points, z_points, indexing="ij")

    yN, yR, WN, WR, omega, f = chn.main_signal_model(Phi, sU, rcs, prior)

    SNRN = np.zeros(grid_flat.shape[0])
    SNRR = np.zeros(grid_flat.shape[0])

    for idx, p in enumerate(grid_flat):
        yN, yR, WN, WR, omega, f = chn.main_signal_model(np.expand_dims(p, axis=0), sU, rcs, prior)
        SNRN[idx] = np.linalg.norm(yN)**2/np.prod(yN.shape) * 2/chn.p_noise
        SNRR[idx] = np.linalg.norm(yR)**2/np.prod(yR.shape) * 2/chn.p_noise
        SNRN[idx] = 10*np.log10(SNRN[idx])
        SNRR[idx] = 10*np.log10(SNRR[idx])

    SNRN = SNRN.reshape((grid_dim[0], grid_dim[1]))
    SNRR = SNRR.reshape((grid_dim[0], grid_dim[1]))

    spec = plt.pcolormesh(x_points, y_points, SNRN.T, cmap="cool", shading="gouraud")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.min(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.max(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.min(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.max(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Non-RIS")
    plt.savefig("results/power_nonRIS.png", dpi=500, bbox_inches="tight")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNRR.T, cmap="cool", shading="gouraud")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.min(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.max(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.min(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.max(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("RIS")
    plt.plot(sR[0], sR[1], "x", color="black", label="RIS")
    plt.savefig("results/power_RIS.png", dpi=500, bbox_inches="tight")
    plt.show()

    contours = plt.contourf(C1[:, :, 0], C2[:, :, 0], SNRN, cmap="cool", levels=6)
    cb = plt.colorbar()
    cb.set_label(label="SNR [dB]")
    plt.clabel(contours, inline = True, fontsize=8, fmt='%d', colors = 'black')
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.min(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.max(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.min(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.max(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Non-RIS")
    plt.savefig("results/contour_power_nonRIS.png", dpi=500, bbox_inches="tight")
    plt.show()

    contours = plt.contourf(C1[:, :, 0], C2[:, :, 0], SNRR, cmap="cool", levels=7)
    cb = plt.colorbar()
    cb.set_label(label="SNR [dB]")
    plt.clabel(contours, inline = True, fontsize=8, fmt='%d', colors = 'black')
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.min(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.max(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.min(ResRegion0, axis=0)[0], np.min(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot([np.max(ResRegion0, axis=0)[0], np.max(ResRegion0, axis=0)[0]], [np.min(ResRegion0, axis=0)[1], np.max(ResRegion0, axis=0)[1]], color="black", ls="dashed")
    plt.plot(sR[0], sR[1], "x", color="black", label="RIS")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("RIS")
    plt.savefig("results/contour_power_RIS.png", dpi=500, bbox_inches="tight")
    plt.show()


def individual_power():
    # =============================================================================
    # Settings
    # =============================================================================
    config_file = "system_config.toml"
    np.random.seed(42)
    pR = np.array([25, 11.5, 2])
    sU = np.array([0, 0, 0, 0, 0, 0])
    pU = sU[:3]
    grid_dim = [30, 16, 1]

    # =============================================================================
    # Run channel
    # =============================================================================
    chn = channel(config_file, False, pR=pR)
    E_s = chn.p_tx
    sigma_eps = np.sqrt(chn.p_noise)
    window = [i + 1 for i in chn.window]
    grid_flat, x_points, y_points, z_points = make_3D_grid(window, grid_dim)

    omega = chn.RIS_random_codebook(T=20) # size=(T, NR)
    f = chn.construct_precoder() # size=(T, NU)

    ###    ###
    SNR_USU = np.zeros(grid_flat.shape[0])
    SNR_URSU = np.zeros(grid_flat.shape[0])
    SNR_USRU = np.zeros(grid_flat.shape[0])
    SNR_URSRU = np.zeros(grid_flat.shape[0])

    for idx, p in enumerate(grid_flat):
        Phi = np.expand_dims(p, 0)
        H_URU, H_URSU, H_USRU, H_URSRU = chn.RIS_channel_matrix(Phi, sU, omega)
        H_USU, H_USSU = chn.nonRIS_channel_matrix(Phi, sU, T=20)

        mu_USU = np.sqrt(E_s)*np.einsum("tnij,tj->tni", H_USU, f)
        SNR_USU[idx] = 10*np.log10(np.linalg.norm(mu_USU)**2/sigma_eps**2)

        mu_URSU = np.sqrt(E_s)*np.einsum("tnij,tj->tni", H_URSU, f)
        SNR_URSU[idx] = 10*np.log10(np.linalg.norm(mu_URSU)**2/sigma_eps**2)

        mu_USRU = np.sqrt(E_s)*np.einsum("tnij,tj->tni", H_USRU, f)
        SNR_USRU[idx] = 10*np.log10(np.linalg.norm(mu_USRU)**2/sigma_eps**2)

        mu_URSRU = np.sqrt(E_s)*np.einsum("tnij,tj->tni", H_URSRU, f)
        SNR_URSRU[idx] = 10*np.log10(np.linalg.norm(mu_URSRU)**2/sigma_eps**2)

    SNR_USU = SNR_USU.reshape((grid_dim[0], grid_dim[1]))
    SNR_URSU = SNR_URSU.reshape((grid_dim[0], grid_dim[1]))
    SNR_USRU = SNR_USRU.reshape((grid_dim[0], grid_dim[1]))
    SNR_URSRU = SNR_URSRU.reshape((grid_dim[0], grid_dim[1]))

    ###    ###
    SNR_USSU = np.zeros(grid_flat.shape[0])

    sims = 4
    d_sigma = np.sqrt(0.5)
    for sim in range(sims):
        for idx, p in enumerate(grid_flat):
            p_neighbor = np.random.multivariate_normal(p, d_sigma*np.eye(3))
            Phi = np.stack((p, p_neighbor))
            H_USU, H_USSU = chn.nonRIS_channel_matrix(Phi, sU)
    
            mu_USSU = np.sqrt(E_s)*np.einsum("tnij,tj->tni", H_USSU, f)
            SNR_USSU[idx] += 10*np.log10(np.linalg.norm(mu_USSU)**2/sigma_eps**2)

    SNR_USSU = (SNR_USSU/sims).reshape((grid_dim[0], grid_dim[1]))

    # =============================================================================
    # Plot power maps
    # =============================================================================

    spec = plt.pcolormesh(x_points, y_points, SNR_USU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-SP-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_URSU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-RIS-SP-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_USRU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-SP-RIS-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_URSRU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-RIS-SP-RIS-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_USSU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-SP-SP-UE")
    plt.show()
    
    # =============================================================================
    # Run channel
    # =============================================================================

    ###    ###
    SNR_URU = np.zeros(grid_flat.shape[0])
    SNR_USU = np.zeros(grid_flat.shape[0])
    SNR_URSU = np.zeros(grid_flat.shape[0])
    SNR_USRU = np.zeros(grid_flat.shape[0])
    SNR_URSRU = np.zeros(grid_flat.shape[0])

    for idx, p in enumerate(grid_flat):
        Phi = np.expand_dims(p, 0)

        pars = chn.ChParams(Phi, pU)
        theta0 = pars["theta0"]
        thetal = pars["thetal"][0]
        phi0 = pars["phi0"]
        phil = pars["phil"][0]
        phi_bounds = (phil[0]-0.2, phil[0]+0.2, phil[1]-0.2, phil[1]+0.2)

        f = chn.construct_precoder(theta0, thetal, T=chn.T2) # size=(T, NU)
        omega = chn.RIS_directional(phi0, phi_bounds) # size=(T, NR)
        W = chn.construct_combiner(theta0) # size=(NU, NU-1)

        H_URU, H_URSU, H_USRU, H_URSRU = chn.RIS_channel_matrix(Phi, sU, omega)
        H_USU, H_USSU = chn.nonRIS_channel_matrix(Phi, sU, T=chn.T2)

        mu_URU = np.sqrt(E_s)*np.einsum("ij,tni->tnj", W.conj(), np.einsum("tnij,tj->tni", H_URU, f))
        SNR_URU[idx] = 10*np.log10(np.linalg.norm(mu_URU)**2/sigma_eps**2)

        mu_USU = np.sqrt(E_s)*np.einsum("ij,tni->tnj", W.conj(), np.einsum("tnij,tj->tni", H_USU, f))
        SNR_USU[idx] = 10*np.log10(np.linalg.norm(mu_USU)**2/sigma_eps**2)

        mu_URSU = np.sqrt(E_s)*np.einsum("ij,tni->tnj", W.conj(), np.einsum("tnij,tj->tni", H_URSU, f))
        SNR_URSU[idx] = 10*np.log10(np.linalg.norm(mu_URSU)**2/sigma_eps**2)

        mu_USRU = np.sqrt(E_s)*np.einsum("ij,tni->tnj", W.conj(), np.einsum("tnij,tj->tni", H_USRU, f))
        SNR_USRU[idx] = 10*np.log10(np.linalg.norm(mu_USRU)**2/sigma_eps**2)

        mu_URSRU = np.sqrt(E_s)*np.einsum("ij,tni->tnj", W.conj(), np.einsum("tnij,tj->tni", H_URSRU, f))
        SNR_URSRU[idx] = 10*np.log10(np.linalg.norm(mu_URSRU)**2/sigma_eps**2)

    SNR_URU = SNR_URU.reshape((grid_dim[0], grid_dim[1]))
    SNR_USU = SNR_USU.reshape((grid_dim[0], grid_dim[1]))
    SNR_URSU = SNR_URSU.reshape((grid_dim[0], grid_dim[1]))
    SNR_USRU = SNR_USRU.reshape((grid_dim[0], grid_dim[1]))
    SNR_URSRU = SNR_URSRU.reshape((grid_dim[0], grid_dim[1]))

    # =============================================================================
    # Plot power maps
    # =============================================================================

    spec = plt.pcolormesh(x_points, y_points, SNR_URU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-RIS-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_USU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-SP-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_URSU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-RIS-SP-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_USRU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-SP-RIS-UE")
    plt.show()

    spec = plt.pcolormesh(x_points, y_points, SNR_URSRU.T, cmap="cool", shading="auto")
    cb = plt.colorbar(spec)
    cb.set_label(label="SNR [dB]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UE-RIS-SP-RIS-UE")
    plt.show()


if __name__ == "__main__":
    # individual_power()
    main_power()



