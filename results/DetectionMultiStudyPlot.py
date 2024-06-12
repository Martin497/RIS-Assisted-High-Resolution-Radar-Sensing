# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 08:22:44 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Make plots for figure 9a and 9b.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")

    # =============================================================================
    # Settings
    # =============================================================================
    folder = "Detection/001"
    with open(f"{folder}/AOAspacings_data.txt", "r") as file:
        temp = file.read()[1:-1].split(",")
        distances = [float(t) for t in temp]
    savenames = [f"{i}" for i in range(len(distances))]
    Nnames = len(savenames)

    RunSpectralDetection = True

    rcs_iters = 1
    fading_sims = 100

    L = 3

    rcsplot = 0

    if RunSpectralDetection is True:
        AUC_spectral_nonRIS, AUC_spectral_RIS, AUC_spectral_joint = np.zeros((Nnames, rcs_iters, fading_sims, L)), np.zeros((Nnames, rcs_iters, fading_sims, L)), np.zeros((Nnames, rcs_iters, fading_sims, L))

    for i, savename in enumerate(savenames):
        with open(f"{folder}/res{savename}_processed.pickle", "rb") as file:
            resDetection = pickle.load(file)

        if RunSpectralDetection is True:
            pD_spectral_nonRIS, pD_spectral_RIS, pD_spectral, pD_spectral_joint = resDetection["pD_spectral_nonRIS"], resDetection["pD_spectral_RIS"], resDetection["pD_spectral"], resDetection["pD_spectral_joint"]
            pFA_spectral_nonRIS, pFA_spectral_RIS, pFA_spectral, pFA_spectral_joint = resDetection["pFA_spectral_nonRIS"], resDetection["pFA_spectral_RIS"], resDetection["pFA_spectral"], resDetection["pFA_spectral_joint"]
            AUC_spectral_nonRIS_, AUC_spectral_RIS_, AUC_spectral_joint_ = resDetection["AUC_spectral_nonRIS"], resDetection["AUC_spectral_RIS"], resDetection["AUC_spectral_joint"]
            for j, l in enumerate(range(1, L+1)):
                AUC_spectral_nonRIS[i, :, :, j] = AUC_spectral_nonRIS_[:, :, L, l]
                AUC_spectral_RIS[i, :, :, j] = AUC_spectral_RIS_[:, :, L, l]
                AUC_spectral_joint[i, :, :, j] = AUC_spectral_joint_[:, :, L, l]

    if RunSpectralDetection is True:
        AUC_spectral_nonRIS_mean = np.mean(AUC_spectral_nonRIS, axis=2)
        AUC_spectral_RIS_mean = np.mean(AUC_spectral_RIS, axis=2)
        AUC_spectral_joint_mean = np.mean(AUC_spectral_joint, axis=2)

    # =============================================================================
    # AUC quantile plot
    # =============================================================================
    if RunSpectralDetection is True:
        for l in range(L):
            plt.plot(distances, AUC_spectral_nonRIS_mean[:, rcsplot, l], color="tab:blue", label="Non-RIS")
            plt.plot(distances, AUC_spectral_RIS_mean[:, rcsplot, l], color="tab:orange", label="RIS")
            plt.plot(distances, AUC_spectral_joint_mean[:, rcsplot, l], color="tab:purple", label="Joint")
            plt.legend()
            plt.ylim(0.5, 1.01)
            plt.show()
