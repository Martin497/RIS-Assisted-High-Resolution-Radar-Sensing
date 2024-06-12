# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:12:58 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Create the plots figure 2 and 4.
"""


import numpy as np
import matplotlib.pyplot as plt
import toml

from TheoreticalAnalysis import TheoreticalInsights


if __name__ == "__main__":
    np.set_printoptions(precision=5, linewidth=np.inf)
    plt.style.use("seaborn-v0_8-whitegrid")
    # =============================================================================
    # Load configuration file
    # =============================================================================
    config_file = "system_config.toml"
    toml_in= toml.load(config_file)
    toml_settings = toml_in["settings"]
    toml_estimation = toml_in["estimation"]

    # =============================================================================
    # Implement setting dictionaries
    # =============================================================================
    np.random.seed(toml_settings["seed"])
    sU = np.array(toml_settings["sU"])

    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    # =============================================================================
    # Expected AUC against coherence (2 targets)
    # =============================================================================
    snr_dB = np.array([10, 20, 30])
    snr = 10**(snr_dB/10)
    coherence = np.linspace(0, 1, 200)

    AUC2 = (1 + snr[None, :]*(1-coherence[:, None])) / (2 + snr[None, :]*(1-coherence[:, None]))

    linestyles=["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for i in range(len(snr_dB)):
        plt.plot(coherence, AUC2[:, i], color="tab:purple", linestyle=linestyles[i], label=f"SNR {snr_dB[i]} dB")
    plt.legend()
    plt.xlabel("Coherence")
    plt.ylabel("Expected AUC")
    plt.show()

    # =============================================================================
    # SNR against coherence for a fixed pFA and pD (2 targets)
    # =============================================================================
    pFA = 1e-03
    pD2 = np.array([0.9, 0.95, 0.99])
    coherence = np.linspace(0, 1, 49, endpoint=False)

    SNR2 = 10*np.log10((np.log(pFA)/np.log(pD2[None, :]) - 1) / (1 - coherence[:, None]))

    linestyles=["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for i in range(len(pD2)):
        plt.plot(coherence, SNR2[:, i], color="tab:purple", linestyle=linestyles[i], label=f"pD {pD2[i]}")
    plt.legend()
    plt.xlabel("Coherence")
    plt.ylabel("Required SNR")
    plt.show()

    # =============================================================================
    # Coherence against AOA spacing Delta
    # =============================================================================
    pos_listN = np.logspace(-3, np.log10(0.15), 90)
    pos_listR = np.logspace(-3, np.log10(0.15), 90)
    M = len(pos_listN)

    NR_list = [[15, 15], [35, 35]]
    T2_list = [8, 18, 50]
    cohR_N_tot = np.zeros((len(NR_list), len(T2_list), M))
    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            toml_settings["NR"] = NR
            toml_settings["T2"] = T2
            mod = TheoreticalInsights(None, **toml_settings)
            cohR_N = np.zeros(M)
            for pos_idx, Delta in enumerate(pos_listR):
                rcs = np.sqrt(np.array([50, 5]))
                Phi_taus = np.array([60, 60])
                az0 = 0.7
                el0 = 0.8
                Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
                Phi_azs = np.array([az0, az0+Delta])
                Phi_els = np.array([el0, el0-Delta])
                Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

                _, cohR_N[pos_idx] = mod.ComputeNormalizedCoherence(Phi, sU, rcs, toml_estimation, False, bounds)
                cohR_N_tot[idx1, idx2] = cohR_N

    NU_list = [[2, 2], [4, 4], [8, 8]]
    cohN_N_tot = np.zeros((len(NU_list), M))
    for idx1, NU in enumerate(NU_list):
        toml_settings["NU"] = NU
        mod = TheoreticalInsights(None, **toml_settings)
        cohN_N = np.zeros(M)
        for pos_idx, Delta in enumerate(pos_listN):
            rcs = np.sqrt(np.array([50, 5]))
            Phi_taus = np.array([60, 60])
            az0 = 0.7
            el0 = 0.8
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array([az0, az0+Delta])
            Phi_els = np.array([el0, el0-Delta])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

            cohN_N[pos_idx], _ = mod.ComputeNormalizedCoherence(Phi, sU, rcs, toml_estimation, False, bounds)
            cohN_N_tot[idx1] = cohN_N

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
        plt.plot(pos_listN, cohN_N_tot[idx1], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Coherence")
    plt.xlim(7.5e-03, 10**(np.log10(0.15)))
    plt.show()


