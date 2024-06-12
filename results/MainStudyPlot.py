# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:57:59 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Make plot for figure 9c.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle

from SensingMetrics import OSPA


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")

    # =============================================================================
    # Settings
    # =============================================================================
    folder = "Main/001"
    with open(f"{folder}/AOAspacings_data.txt", "r") as file:
        temp = file.read()[1:-1].split(",")
        distances = [float(t) for t in temp]
    savenames = [f"res{i}" for i in range(len(distances))]

    rcs_iters = 1
    fading_sims = 15
    L = 3

    pFA_idx = 3
    th_comp = 100

    rcsplot = 0

    noise_sims = 1

    th_spectral = 200
    confidence_level = np.linspace(0, 1, th_spectral)

    p = 2
    c = 3

    OSPA_joint_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    OSPAN_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    OSPAR_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    counter = 0
    for savename in savenames:
        with open(f"{folder}/{savename}.pickle", "rb") as file:
            res = pickle.load(file)
    
        # =============================================================================
        # Setup
        # =============================================================================
        OSPAN = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))
        OSPAR = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))
        OSPA_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))

        # =============================================================================
        # Load data and compute KPIs
        # =============================================================================
        for idx1 in range(rcs_iters):
            for idx2 in range(fading_sims):
                output = res[f"{idx1}"][f"{idx2}"]
                for idx3 in range(1, L+1):
                    Phi = res[f"{idx1}"][f"{idx2}"][f"{idx3}"]["0"]["Phi"]
                    for idx4 in range(noise_sims):
                        output = res[f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"]
                        PosEstN_, PosEstR_, PosEst_joint_ = output["PosEstN"], output["PosEstR"], output["PosEst"]
                        OSPAN[idx1, idx2, idx3, idx4] = OSPA(PosEstN_, Phi, p, c)
                        OSPAR[idx1, idx2, idx3, idx4] = OSPA(PosEstR_, Phi, p, c)
                        OSPA_joint[idx1, idx2, idx3, idx4] = OSPA(PosEst_joint_, Phi, p, c)

        # =============================================================================
        # OSPA cdf
        # =============================================================================
        OSPA_x = np.linspace(0, c, 100)
        linestyles = ["solid", "dashed", "dotted"]
        for l in range(1, L+1):
            print(np.mean(OSPA_joint[0, :, l, :]), np.mean(OSPAR[0, :, l, :]), np.mean(OSPAN[0, :, l, :]))
            OSPA_yJ = [np.sum(OSPA_joint[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in OSPA_x]
            OSPA_yR = [np.sum(OSPAR[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in OSPA_x]
            OSPA_yN = [np.sum(OSPAN[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in OSPA_x]
            plt.plot(OSPA_x, OSPA_yJ, color="tab:purple",label="Joint")
            plt.plot(OSPA_x, OSPA_yR, color="tab:orange",label="RIS")
            plt.plot(OSPA_x, OSPA_yN, color="tab:blue", label="Non-RIS")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("Pr(OSPA $\leq$ x)")
            plt.title(f"Number of targets: {l}")
            plt.ylim(-0.03, 1.03)
            plt.show()

        OSPA_joint_tot[counter] = OSPA_joint[0, :, :, :]
        OSPAN_tot[counter] = OSPAN[0, :, :, :]
        OSPAR_tot[counter] = OSPAR[0, :, :, :]
        counter += 1

    plt.plot(distances, np.mean(OSPA_joint_tot[:, :, -1, :], axis=(1,2)), color="tab:purple", label="Joint")
    plt.plot(distances, np.mean(OSPAR_tot[:, :, -1, :], axis=(1,2)), color="tab:orange", label="RIS")
    plt.plot(distances, np.mean(OSPAN_tot[:, :, -1, :], axis=(1,2)), color="tab:blue", label="Non-RIS")
    plt.legend()
    plt.ylabel("OSPA")
    plt.xlabel("AOA spacing")
    plt.show()
