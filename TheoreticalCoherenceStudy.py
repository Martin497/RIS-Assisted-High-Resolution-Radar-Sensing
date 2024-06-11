# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:12:58 2024

@author: BX98LW
"""


import numpy as np
import matplotlib.pyplot as plt
import toml

# import tikzplotlib

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

    # bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
    #           "theta_bounds": np.array([0.57, 0.87, 0.65, 0.95]),
    #           "tau_bar_bounds": np.array([1.13e-07, 1.32e-07]),
    #           "phi_bounds": np.array([0.10, 0.78, 0.55, 1.16])}
    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    # =============================================================================
    # Analyze normalized coherence
    # =============================================================================
    pos_listN = np.logspace(-3, np.log10(0.6), 90)
    pos_listR = np.logspace(-3, np.log10(0.15), 90)
    M = len(pos_listN)

    pFA = 1e-07

    NR_list = [[15, 15], [35, 35]]
    T2_list = [8, 18, 50]
    cohR_N_tot = np.zeros((len(NR_list), len(T2_list), M))
    pdr_tot = np.zeros((len(NR_list), len(T2_list), M, 2))
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

                _, _, _, pd1r, pd2r, _ = mod.ComputeExpectedDetectionProbability(pFA, Phi, sU, rcs, toml_estimation, False, bounds)
                pdr_tot[idx1, idx2, pos_idx, 0] = pd1r
                pdr_tot[idx1, idx2, pos_idx, 1] = pd2r

    NU_list = [[2, 2], [4, 4], [8, 8]]
    cohN_N_tot = np.zeros((len(NU_list), M))
    pdn_tot = np.zeros((len(NU_list), M, 3))
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

            pd1n, pd2n, _, _, _, _ = mod.ComputeExpectedDetectionProbability(pFA, Phi, sU, rcs, toml_estimation, False, bounds)
            pdn_tot[idx1, pos_idx, 0] = pd1n
            pdn_tot[idx1, pos_idx, 1] = pd2n

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
            # if idx1 == 0 and idx2 == 0:
            #     plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
            # elif idx1 == 0 and idx2 == 2:
            #     plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
            # elif idx1 == 1 and idx2 == 0:
            #     plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
            # elif idx1 == 1 and idx2 == 2:
            #     plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
            # elif idx1 == 2:
            #     plt.plot(pos_listR, cohR_N_tot[idx1, idx2], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
        plt.plot(pos_listN, cohN_N_tot[idx1], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Coherence")
    # plt.axvline(0.17)
    plt.xlim(7.5e-03, 10**(np.log10(0.15)))
    # tikzplotlib.save("results/Theoretical/coherence.tex")
    # plt.savefig("results/Theoretical/coherence.png", dpi=500, bbox_inches="tight")
    plt.show()

    with open("results/Theoretical/coherence.txt", "w") as file:
        color_list = ["color3", "color5"]
        linestyle_list = ["dotted", "dashed", "solid"]
        for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
            for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
                file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {col}, {ls}]\n")
                file.write("table {%\n")
                for x, y in zip(pos_listR, cohR_N_tot[idx1, idx2]):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("}; \\label{plot:RIS_T"+f"{T2//2}_NR{NR[0]}"+"}\n")
        for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
            file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {ls}]\n")
            file.write("table {%\n")
            for x, y in zip(pos_listN, cohN_N_tot[idx1]):
                file.write(f"{x:.4f}  {y:.4f}\n")
            file.write("}; \\label{plot:nonRIS_NU"+f"{NU[0]}"+"}\n")


    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(cohR_N_tot[idx1, idx2], pdr_tot[idx1, idx2, :, 0], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
        plt.plot(cohN_N_tot[idx1], pdn_tot[idx1, :, 0], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xlabel("Coherence")
    plt.ylabel("Detection probability")
    plt.xlim(0, 1)
    plt.legend()
    # tikzplotlib.save("results/Theoretical/Epd_coherence_Target1.tex")
    # plt.savefig("results/Theoretical/Epd_coherence_Target1.png", dpi=500, bbox_inches="tight")
    plt.show()

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(cohR_N_tot[idx1, idx2], pdr_tot[idx1, idx2, :, 1], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    for idx1, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
        plt.plot(cohN_N_tot[idx1], pdn_tot[idx1, :, 1], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xlabel("Coherence")
    plt.ylabel("Detection probability")
    plt.xlim(0, 1)
    plt.legend()
    # tikzplotlib.save("results/Theoretical/Epd_coherence_Target2.tex")
    # plt.savefig("results/Theoretical/Epd_coherence_Target2.png", dpi=500, bbox_inches="tight")
    plt.show()


    # =============================================================================
    # Expected detection probability against coherence (2 targets)
    # =============================================================================
    pFA = 1e-07
    snr_dB = np.array([10, 20, 30])
    snr = 10**(snr_dB/10)
    coherence = np.linspace(0, 1, 200)

    pd2 = np.exp(np.log(pFA)/(snr[None, :]*(1-coherence[:, None])+1))

    linestyles=["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for i in range(len(snr_dB)):
        plt.plot(coherence, pd2[:, i], color="tab:purple", linestyle=linestyles[i], label=f"SNR {snr_dB[i]} dB")
    plt.legend()
    plt.xlabel("Coherence")
    plt.ylabel("Expected detection probability")
    plt.show()

    # =============================================================================
    # Expected AUC against coherence (2 targets)
    # =============================================================================
    AUC2 = (1 + snr[None, :]*(1-coherence[:, None])) / (2 + snr[None, :]*(1-coherence[:, None]))

    linestyles=["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for i in range(len(snr_dB)):
        plt.plot(coherence, AUC2[:, i], color="tab:purple", linestyle=linestyles[i], label=f"SNR {snr_dB[i]} dB")
    plt.legend()
    plt.xlabel("Coherence")
    plt.ylabel("Expected AUC")
    plt.show()

    with open("results/Theoretical/AUC_vs_coherence.txt", "w") as file:
        for i in range(len(snr_dB)):
            file.write(f"\\addplot [semithick, {'color3'}, {linestyles[i]}]\n")
            file.write("table {%\n")
            for x, y in zip(coherence, AUC2[:, i]):
                file.write(f"{x:.4f}  {y:.4f}\n")
            file.write("}; \\addlegendentry{SNR = "+f"{snr_dB[i]}"+" dB}\n")

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

    with open("results/Theoretical/SNR_vs_coherence.txt", "w") as file:
        for i in range(len(pD2)):
            file.write(f"\\addplot [semithick, {'color3'}, {linestyles[i]}]\n")
            file.write("table {%\n")
            for x, y in zip(coherence, SNR2[:, i]):
                file.write(f"{x:.4f}  {y:.4f}\n")
            file.write("}; \\addlegendentry{pD = "+f"{pD2[i]}"+" dB}\n")

    # =============================================================================
    # SNR against coherence for a fixed AUC (2 targets)
    # =============================================================================
    AUC2 = np.array([0.8, 0.95, 0.99])
    coherence = np.linspace(0, 1, 199, endpoint=False)

    SNR2 = 10*np.log10((1 - 2*AUC2[None, :]) / ((AUC2[None, :] - 1)*(1 - coherence[:, None])))

    linestyles=["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for i in range(len(AUC2)):
        plt.plot(coherence, SNR2[:, i], color="tab:purple", linestyle=linestyles[i], label=f"AUC {AUC2[i]}")
    plt.legend()
    plt.xlabel("Coherence")
    plt.ylabel("Required SNR")
    plt.show()


