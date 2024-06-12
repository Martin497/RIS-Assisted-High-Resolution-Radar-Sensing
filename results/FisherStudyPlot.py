# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:42:35 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Create the plot figure 8.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle

def quantiles(y):
    y_05quantile = np.quantile(y, q=0.05, axis=1)
    y_25quantile = np.quantile(y, q=0.25, axis=1)
    y_50quantile = np.quantile(y, q=0.50, axis=1)
    y_75quantile = np.quantile(y, q=0.75, axis=1)
    y_95quantile = np.quantile(y, q=0.95, axis=1)
    return y_05quantile, y_25quantile, y_50quantile, y_75quantile, y_95quantile

def functional_boxplot(x, nonRIS, RIS, combined, ylabel, title, savename=None):
    """
    """
    nonRIS_quantile = quantiles(nonRIS)
    RIS_quantile = quantiles(RIS)
    if combined is not None:
        combined_quantile = quantiles(combined)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, nonRIS_quantile[2], color="navy", label="Non-RIS")
    ax.fill_between(x, nonRIS_quantile[0], nonRIS_quantile[1], color="cornflowerblue")
    ax.fill_between(x, nonRIS_quantile[1], nonRIS_quantile[2], color="tab:blue")
    ax.fill_between(x, nonRIS_quantile[2], nonRIS_quantile[3], color="tab:blue")
    ax.fill_between(x, nonRIS_quantile[3], nonRIS_quantile[4], color="cornflowerblue")

    plt.plot(x, RIS_quantile[2], color="tab:orange", label="RIS")
    ax.fill_between(x, RIS_quantile[0], RIS_quantile[1], color="navajowhite")
    ax.fill_between(x, RIS_quantile[1], RIS_quantile[2], color="gold")
    ax.fill_between(x, RIS_quantile[2], RIS_quantile[3], color="gold")
    ax.fill_between(x, RIS_quantile[3], RIS_quantile[4], color="navajowhite")

    if combined is not None:
        plt.plot(x, combined_quantile[2], color="indigo", label="Combined")
        ax.fill_between(x, combined_quantile[0], combined_quantile[1], color="tab:purple")
        ax.fill_between(x, combined_quantile[1], combined_quantile[2], color="darkviolet")
        ax.fill_between(x, combined_quantile[2], combined_quantile[3], color="darkviolet")
        ax.fill_between(x, combined_quantile[3], combined_quantile[4], color="tab:purple")

    plt.yscale("log")
    plt.xlabel("AOA spacing [rad]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")

    # =============================================================================
    # Settings
    # =============================================================================
    savename = "001"
    folder = f"Fisher/{savename}"
    distances = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    Ndists = len(distances)

    rcsplot = 0
    fading_sims = 360

    L = 3
    d = 3

    with open(f"{folder}/FisherResults{savename}.pickle", "rb") as file:
        resFisher = pickle.load(file)

    PEB_USU = np.zeros((Ndists, L, fading_sims))
    PEB_USRU = np.zeros((Ndists, L, fading_sims))
    PEB_combined = np.zeros((Ndists, L, fading_sims))
    DEB_tau = np.zeros((Ndists, L, fading_sims))
    DEB_tau_bar = np.zeros((Ndists, L, fading_sims))
    OEB_ThetaAz = np.zeros((Ndists, L, fading_sims))
    OEB_ThetaEl = np.zeros((Ndists, L, fading_sims))
    OEB_Theta = np.zeros((Ndists, L, fading_sims))
    OEB_PhiAz = np.zeros((Ndists, L, fading_sims))
    OEB_PhiEl = np.zeros((Ndists, L, fading_sims))
    OEB_Phi = np.zeros((Ndists, L, fading_sims))
    for i, key in enumerate(resFisher):
        for j in range(fading_sims):
            data = resFisher[f"{key}"][f"{rcsplot}"][f"{j}"]
            CRLB_USU, CRLBx_USU = data["CRLB_USU"], data["CRLBx_USU"]
            CRLB_USRU, CRLBx_USRU = data["CRLB_USRU"], data["CRLBx_USRU"]
            CRLB_combined, CRLBx_combined = data["CRLB_combined"], data["CRLBx_combined"]

            for l in range(L):
                PEB_USU[i, l, j] = np.sqrt(np.trace(CRLBx_USU[l*d:(l+1)*d, l*d:(l+1)*d]))
                DEB_tau[i, l, j] = np.sqrt(CRLB_USU[l, l])
                OEB_ThetaAz[i, l, j] = np.sqrt(CRLB_USU[L+l, L+l])
                OEB_ThetaEl[i, l, j] = np.sqrt(CRLB_USU[2*L+l, 2*L+l])
                OEB_Theta[i, l, j] = np.sqrt(CRLB_USU[L+l, L+l]+CRLB_USU[2*L+l, 2*L+l])

                PEB_USRU[i, l, j] = np.sqrt(np.trace(CRLBx_USRU[l*d:(l+1)*d, l*d:(l+1)*d]))
                DEB_tau_bar[i, l, j] = np.sqrt(CRLB_USRU[l, l])
                OEB_PhiAz[i, l, j] = np.sqrt(CRLB_USRU[L+l, L+l])
                OEB_PhiEl[i, l, j] = np.sqrt(CRLB_USRU[2*L+l, 2*L+l])
                OEB_Phi[i, l, j] = np.sqrt(CRLB_USRU[L+l, L+l]+CRLB_USRU[2*L+l, 2*L+l])

                PEB_combined[i, l, j] = np.sqrt(np.trace(CRLBx_combined[l*d:(l+1)*d, l*d:(l+1)*d]))

    distances_idx = 0
    PEBx = np.linspace(0, 1.4, 100)
    linestyles = ["solid", "dashed", "dotted"]
    for l, ls in enumerate(linestyles):
        cdf_PEB_nonRIS = [np.sum(PEB_USU[distances_idx, l, :] <= x)/fading_sims for x in PEBx]
        cdf_PEB_RIS = [np.sum(PEB_USRU[distances_idx, l, :] <= x)/fading_sims for x in PEBx]
        cdf_PEB_joint = [np.sum(PEB_combined[distances_idx, l, :] <= x)/fading_sims for x in PEBx]

        plt.plot(PEBx, cdf_PEB_joint, color="tab:purple", linestyle=ls, label=f"Joint, Target {l+1}")
        plt.plot(PEBx, cdf_PEB_RIS, color="tab:orange", linestyle=ls, label=f"RIS, Target {l+1}")
        plt.plot(PEBx, cdf_PEB_nonRIS, color="tab:blue", linestyle=ls, label=f"Non-RIS, Target {l+1}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Pr(PEB $\leq$ x)")
    plt.xlim(0, 1.4)
    plt.show()

    # =============================================================================
    # DEB cdf
    # =============================================================================
    distances_idx = 0
    DEBx = np.linspace(0, 1.6, 100)
    linestyles = ["solid", "dashed", "dotted"]
    for l, ls in enumerate(linestyles):
        cdf_DEB_nonRIS = [np.sum(DEB_tau[distances_idx, l, :] <= x)/fading_sims for x in DEBx]
        cdf_DEB_RIS = [np.sum(DEB_tau_bar[distances_idx, l, :] <= x)/fading_sims for x in DEBx]

        plt.plot(DEBx, cdf_DEB_nonRIS, color="tab:blue", linestyle=ls, label=f"Non-RIS, Target {l+1}")
        plt.plot(DEBx, cdf_DEB_RIS, color="tab:orange", linestyle=ls, label=f"RIS, Target {l+1}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Pr(DEB $\leq$ x)")
    plt.xlim(0, 1.6)
    plt.show()

    # =============================================================================
    # AEB cdf
    # =============================================================================
    distances_idx = 0
    AEBx = np.linspace(0, 0.1, 100)
    linestyles = ["solid", "dashed", "dotted"]
    for l, ls in enumerate(linestyles):
        cdf_AEB_nonRIS = [np.sum(OEB_Theta[distances_idx, l, :] <= x)/fading_sims for x in AEBx]
        cdf_AEB_RIS = [np.sum(OEB_Phi[distances_idx, l, :] <= x)/fading_sims for x in AEBx]

        plt.plot(AEBx, cdf_AEB_nonRIS, color="tab:blue", linestyle=ls, label=f"Non-RIS, Target {l+1}")
        plt.plot(AEBx, cdf_AEB_RIS, color="tab:orange", linestyle=ls, label=f"RIS, Target {l+1}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Pr(AEB $\leq$ x)")
    plt.xlim(0, 0.1)
    plt.show()
