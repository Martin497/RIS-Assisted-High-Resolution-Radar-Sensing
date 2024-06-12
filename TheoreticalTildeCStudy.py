# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:39:41 2024

Create the plot figure 5.
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
    # Analyze normalized coherence
    # =============================================================================
    pos_listN = np.logspace(-3, np.log10(0.15), 90)
    pos_listR = np.logspace(-3, np.log10(0.15), 90)
    M = len(pos_listN)

    pFA = 1e-07

    NR_list = [[15, 15], [35, 35]]
    T2_list = [8, 18, 50]
    TildeCR_tot = np.zeros((len(NR_list), len(T2_list), M))
    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            toml_settings["NR"] = NR
            toml_settings["T2"] = T2
            mod = TheoreticalInsights(None, **toml_settings)
            TildeCR = np.zeros(M)
            for pos_idx, Delta in enumerate(pos_listR):
                rcs = np.sqrt(np.array([50, 5, 0.5]))
                Phi_taus = np.array([60, 60, 60])
                az0 = 0.7
                el0 = 0.8
                Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
                Phi_azs = np.array([az0, az0+Delta, az0-Delta])
                Phi_els = np.array([el0, el0-Delta, az0+Delta])
                Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

                _, TildeCR[pos_idx] = mod.ComputeTildeC(Phi, sU, rcs, toml_estimation, False, bounds)
                TildeCR_tot[idx1, idx2] = TildeCR

    NU_list = [[2, 2], [4, 4], [8, 8]]
    TildeCN_tot = np.zeros((len(NU_list), M))
    for idx1, NU in enumerate(NU_list):
        toml_settings["NU"] = NU
        mod = TheoreticalInsights(None, **toml_settings)
        TildeCN = np.zeros(M)
        for pos_idx, Delta in enumerate(pos_listN):
            rcs = np.sqrt(np.array([50, 5]))
            Phi_taus = np.array([60, 60])
            az0 = 0.7
            el0 = 0.8
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array([az0, az0+Delta])
            Phi_els = np.array([el0, el0-Delta])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

            TildeCN[pos_idx], _ = mod.ComputeNormalizedCoherence(Phi, sU, rcs, toml_estimation, False, bounds)
            TildeCN_tot[idx1] = TildeCN

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, TildeCR_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
        plt.plot(pos_listN, TildeCN_tot[idx1], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Coherence")
    plt.xlim(7.5e-03, 10**(np.log10(0.15)))
    plt.show()
