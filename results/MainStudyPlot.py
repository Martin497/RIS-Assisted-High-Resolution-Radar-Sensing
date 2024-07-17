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

import os
import sys
if os.path.abspath("..") not in sys.path:
    sys.path.append(os.path.abspath(".."))

from SensingMetrics import OSPA, GOSPA


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
    c = 5

    GOSPA_joint_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    GOSPAN_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    GOSPAR_tot = np.zeros((len(distances), fading_sims, L+1, noise_sims))
    counter = 0
    for savename in savenames:
        with open(f"{folder}/{savename}.pickle", "rb") as file:
            res = pickle.load(file)
    
        # =============================================================================
        # Setup
        # =============================================================================
        GOSPAN = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))
        GOSPAR = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))
        GOSPA_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims))

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
                        GOSPAN[idx1, idx2, idx3, idx4] = GOSPA(PosEstN_, Phi, p, c, 2)
                        GOSPAR[idx1, idx2, idx3, idx4] = GOSPA(PosEstR_, Phi, p, c, 2)
                        GOSPA_joint[idx1, idx2, idx3, idx4] = GOSPA(PosEst_joint_, Phi, p, c, 2)

        # =============================================================================
        # GOSPA cdf
        # =============================================================================
        GOSPA_x = np.linspace(0, c, 100)
        linestyles = ["solid", "dashed", "dotted"]
        # for l in range(1, L+1):
        #     print(np.mean(GOSPA_joint[0, :, l, :]), np.mean(GOSPAR[0, :, l, :]), np.mean(GOSPAN[0, :, l, :]))
        #     GOSPA_yJ = [np.sum(GOSPA_joint[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in GOSPA_x]
        #     GOSPA_yR = [np.sum(GOSPAR[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in GOSPA_x]
        #     GOSPA_yN = [np.sum(GOSPAN[0, :, l, :] <= x)/(fading_sims*noise_sims) for x in GOSPA_x]
        #     plt.plot(GOSPA_x, GOSPA_yJ, color="tab:purple",label="Joint")
        #     plt.plot(GOSPA_x, GOSPA_yR, color="tab:orange",label="RIS")
        #     plt.plot(GOSPA_x, GOSPA_yN, color="tab:blue", label="Non-RIS")
        #     plt.legend()
        #     plt.xlabel("x")
        #     plt.ylabel("Pr(GOSPA $\leq$ x)")
        #     plt.title(f"Number of targets: {l}")
        #     plt.ylim(-0.03, 1.03)
        #     plt.show()

        GOSPA_joint_tot[counter] = GOSPA_joint[0, :, :, :]
        GOSPAN_tot[counter] = GOSPAN[0, :, :, :]
        GOSPAR_tot[counter] = GOSPAR[0, :, :, :]
        counter += 1

    plt.plot(distances, np.mean(GOSPA_joint_tot[:, :, -1, :], axis=(1,2)), color="tab:purple", label="Joint")
    plt.plot(distances, np.mean(GOSPAR_tot[:, :, -1, :], axis=(1,2)), color="tab:orange", label="RIS")
    plt.plot(distances, np.mean(GOSPAN_tot[:, :, -1, :], axis=(1,2)), color="tab:blue", label="Non-RIS")
    plt.xscale("log")
    plt.legend()
    plt.ylabel("GOSPA")
    plt.xlabel("AOA spacing")
    plt.show()
