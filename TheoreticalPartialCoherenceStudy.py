# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:43:14 2024

@author: BX98LW
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

    # bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
    #           "theta_bounds": np.array([0.57, 0.87, 0.65, 0.95]),
    #           "tau_bar_bounds": np.array([1.13e-07, 1.32e-07]),
    #           "phi_bounds": np.array([0.10, 0.78, 0.55, 1.16])}
    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False
    
    # =============================================================================
    # Analyze Fisher inner products
    # =============================================================================
    # pos_list = np.logspace(-4, np.log10(0.15), 100)
    # pos_listN = np.linspace(0.01, 0.15, 50)
    # pos_listR = np.linspace(0.01, 0.15, 50)
    pos_listN = np.logspace(-3, np.log10(0.6), 90)
    pos_listR = np.logspace(-3, np.log10(0.15), 90)
    MN = len(pos_listN)
    MR = len(pos_listR)

    NR_list = [[15, 15], [35, 35]]
    T2_list = [8, 18, 50]
    NU_list = [[2, 2], [4, 4], [8, 8]]
    aU1aU2_tot = np.zeros((len(NU_list), MN), dtype=np.complex128)
    aU1deraU2_tot = np.zeros((len(NU_list), MN), dtype=np.complex128)
    aU2deraU1_tot = np.zeros((len(NU_list), MN), dtype=np.complex128)
    deraU1deraU2_tot = np.zeros((len(NU_list), MN), dtype=np.complex128)
    nu1nu2_tot = np.zeros((len(NR_list), len(T2_list), MR), dtype=np.complex128)
    nu1dernu2_tot = np.zeros((len(NR_list), len(T2_list), MR), dtype=np.complex128)
    nu2dernu1_tot = np.zeros((len(NR_list), len(T2_list), MR), dtype=np.complex128)
    dernu1dernu2_tot = np.zeros((len(NR_list), len(T2_list), MR), dtype=np.complex128)
    for pos_idx, Delta in enumerate(pos_listR):
        rcs = np.sqrt(np.array([50, 5]))
        Phi_taus = np.array([60, 60])
        az0 = 0.7
        el0 = 0.8
        Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
        Phi_azs = np.array([az0, az0+Delta])
        Phi_els = np.array([el0, el0-Delta])
        Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

        for idx1, NR in enumerate(NR_list):
            for idx2, T2 in enumerate(T2_list):
                toml_settings["NR"] = NR
                toml_settings["T2"] = T2
                mod = TheoreticalInsights(None, **toml_settings)
                _, _, _, _, nu1, nu2, dernu1, dernu2 \
                    = mod.ComputeArrayResponseInnerProducts(Phi, sU, rcs, toml_estimation, False, bounds)

                nu1nu2 = np.dot(nu1.conj(), nu2)/(np.linalg.norm(nu1)*np.linalg.norm(nu2))
                nu1dernu2 = np.dot(nu1.conj(), dernu2)/(np.linalg.norm(nu1)*np.linalg.norm(dernu2))
                nu2dernu1 = np.dot(nu2.conj(), dernu1)/(np.linalg.norm(nu2)*np.linalg.norm(dernu1))
                dernu1dernu2 = np.dot(dernu1.conj(), dernu2)/(np.linalg.norm(dernu1)*np.linalg.norm(dernu2))

                nu1nu2_tot[idx1, idx2, pos_idx] = nu1nu2
                nu1dernu2_tot[idx1, idx2, pos_idx] = nu1dernu2
                nu2dernu1_tot[idx1, idx2, pos_idx] = nu2dernu1
                dernu1dernu2_tot[idx1, idx2, pos_idx] = dernu1dernu2

    for pos_idx, Delta in enumerate(pos_listN):
        rcs = np.sqrt(np.array([50, 5]))
        Phi_taus = np.array([60, 60])
        az0 = 0.7
        el0 = 0.8
        Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
        Phi_azs = np.array([az0, az0+Delta])
        Phi_els = np.array([el0, el0-Delta])
        Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

        for idx3, NU in enumerate(NU_list):
            toml_settings["NU"] = NU
            mod = TheoreticalInsights(None, **toml_settings)
            aU1, aU2, deraU1, deraU2, _, _, _, _ \
                = mod.ComputeArrayResponseInnerProducts(Phi, sU, rcs, toml_estimation, False, bounds)

            aU1aU2 = np.dot(aU1.conj(), aU2)/(np.linalg.norm(aU1)*np.linalg.norm(aU2))
            aU1deraU2 = np.dot(aU1.conj(), deraU2)/(np.linalg.norm(aU1)*np.linalg.norm(deraU2))
            aU2deraU1 = np.dot(aU2.conj(), deraU1)/(np.linalg.norm(aU2)*np.linalg.norm(deraU1))
            deraU1deraU2 = np.dot(deraU1.conj(), deraU2)/(np.linalg.norm(deraU1)*np.linalg.norm(deraU2))

            aU1aU2_tot[idx3, pos_idx] = aU1aU2
            aU1deraU2_tot[idx3, pos_idx] = aU1deraU2
            aU2deraU1_tot[idx3, pos_idx] = aU2deraU1
            deraU1deraU2_tot[idx3, pos_idx] = deraU1deraU2

    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # plt.plot(pos_list, np.abs(aU1aU2_tot)**2, color="midnightblue")
    # plt.plot(pos_list, np.abs(nu1nu2_tot)**2, color="darkorange")
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.title("1 2")
    # plt.show()

    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # plt.plot(pos_list, np.abs(aU1deraU2_tot)**2, color="midnightblue")
    # plt.plot(pos_list, np.abs(nu1dernu2_tot)**2, color="darkorange")
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.title("1 der2")
    # plt.show()

    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # plt.plot(pos_list, np.abs(aU2deraU1_tot)**2, color="midnightblue")
    # plt.plot(pos_list, np.abs(nu2dernu1_tot)**2, color="darkorange")
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.title("2 der1")
    # plt.show()

    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # plt.plot(pos_list, np.abs(deraU1deraU2_tot)**2, color="midnightblue")
    # plt.plot(pos_list, np.abs(dernu1dernu2_tot)**2, color="darkorange")
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.title("der1 der2")
    # plt.show()

    # counter = 0
    # color_list = ["darkorange", "tab:purple", "tab:red", "gold"]
    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # for idx1, NR in enumerate(NR_list):
    #     for idx2, T2 in enumerate(T2_list):
    #         if not (idx1 == 0 and idx2 == 0):
    #             plt.plot(pos_listR, np.abs(nu1nu2_tot[idx1, idx2])**2, color=color_list[counter], linestyle="solid", marker=".", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2} 1 2")
    #             plt.plot(pos_listR, np.abs(nu1dernu2_tot[idx1, idx2])**2, color=color_list[counter], linestyle="dashed", marker="x", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2} 1 der2")
    #             plt.plot(pos_listR, np.abs(nu2dernu1_tot[idx1, idx2])**2, color=color_list[counter], linestyle="dotted", marker="+", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2} 2 der1")
    #             plt.plot(pos_listR, np.abs(dernu1dernu2_tot[idx1, idx2])**2, color=color_list[counter], linestyle="dashdot", marker="*", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2} der1 der2")
    #             counter += 1
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.legend(loc="upper right")
    # # plt.savefig("results/Theoretical/InnerProductRIS.png", dpi=500, bbox_inches="tight")
    # plt.show()

    # counter = 0
    # color_list = ["midnightblue", "tab:green", "tab:olive"]
    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # for idx3, NU in enumerate(NU_list):
    #     plt.plot(pos_listN, np.abs(aU1aU2_tot[idx3])**2, color=color_list[counter], linestyle="solid", marker=".", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]} 1 2")
    #     plt.plot(pos_listN, np.abs(aU1deraU2_tot[idx3])**2, color=color_list[counter], linestyle="dashed", marker="x", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]} 1 der2")
    #     plt.plot(pos_listN, np.abs(aU2deraU1_tot[idx3])**2, color=color_list[counter], linestyle="dotted", marker="+", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]} 2 der1")
    #     plt.plot(pos_listN, np.abs(deraU1deraU2_tot[idx3])**2, color=color_list[counter], linestyle="dashdot", marker="*", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]} der1 der2")
    #     counter += 1
    # plt.ylim(-0.01, 1.01)
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("Coherence")
    # plt.legend(loc="upper right")
    # # plt.savefig("results/Theoretical/InnerProductNonRIS.png", dpi=500, bbox_inches="tight")
    # plt.show()

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, np.mean(np.array([np.abs(nu1nu2_tot[idx1, idx2])**2, np.abs(nu1dernu2_tot[idx1, idx2])**2, np.abs(nu2dernu1_tot[idx1, idx2])**2, np.abs(dernu1dernu2_tot[idx1, idx2])**2]), axis=0),
                     color=col, linestyle=ls, marker="", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    linestyle_list = ["solid", "dashed", "dotted"]
    for idx3, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
        plt.plot(pos_listN, np.mean(np.array([np.abs(aU1aU2_tot[idx3])**2, np.abs(aU1deraU2_tot[idx3])**2, np.abs(aU2deraU1_tot[idx3])**2, np.abs(deraU1deraU2_tot[idx3])**2]), axis=0),
                 color="midnightblue", linestyle=ls, marker="", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.ylim(-0.01, 1.01)
    plt.xlim(7.5e-03, 0.15)
    # plt.xlim(pos_listR[0]-0.001, pos_listR[-1]+0.001)
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Average coherence")
    plt.legend(loc="lower left")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.savefig("results/Theoretical/InnerProductAggregated.png", dpi=500, bbox_inches="tight")
    plt.show()

    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, np.max(np.array([np.abs(nu1nu2_tot[idx1, idx2])**2, np.abs(nu1dernu2_tot[idx1, idx2])**2, np.abs(nu2dernu1_tot[idx1, idx2])**2, np.abs(dernu1dernu2_tot[idx1, idx2])**2]), axis=0),
                     color=col, linestyle=ls, marker="", label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    linestyle_list = ["solid", "dashed", "dotted"]
    for idx3, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
        plt.plot(pos_listN, np.max(np.array([np.abs(aU1aU2_tot[idx3])**2, np.abs(aU1deraU2_tot[idx3])**2, np.abs(aU2deraU1_tot[idx3])**2, np.abs(deraU1deraU2_tot[idx3])**2]), axis=0),
                 color="midnightblue", linestyle=ls, marker="", label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.ylim(-0.01, 1.01)
    plt.xlim(7.5e-03, 0.15)
    # plt.xlim(pos_listR[0]-0.001, pos_listR[-1]+0.001)
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Maximum coherence")
    plt.legend(loc="lower left")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.savefig("results/Theoretical/InnerProductMax.png", dpi=500, bbox_inches="tight")
    plt.show()

    with open("results/Theoretical/InnerProductMax.txt", "w") as file:
        color_list = ["color3", "color5"]
        linestyle_list = ["dotted", "dashed", "solid"]
        for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
            for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
                file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {col}, {ls}]\n")
                file.write("table {%\n")
                partial_coh_max = np.max(np.array([np.abs(nu1nu2_tot[idx1, idx2])**2, np.abs(nu1dernu2_tot[idx1, idx2])**2, np.abs(nu2dernu1_tot[idx1, idx2])**2, np.abs(dernu1dernu2_tot[idx1, idx2])**2]), axis=0)
                for x, y in zip(pos_listR, partial_coh_max):
                    file.write(f"{x:.4f}  {y:.4f}\n")
                file.write("}; \\label{plot:RIS_T"+f"{T2//2}_NR{NR[0]}"+"}\n")
        for idx3, (ls, NU) in enumerate(zip(np.flip(linestyle_list), NU_list)):
            file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {ls}]\n")
            file.write("table {%\n")
            partial_coh_max = np.max(np.array([np.abs(aU1aU2_tot[idx3])**2, np.abs(aU1deraU2_tot[idx3])**2, np.abs(aU2deraU1_tot[idx3])**2, np.abs(deraU1deraU2_tot[idx3])**2]), axis=0)
            for x, y in zip(pos_listN, partial_coh_max):
                file.write(f"{x:.4f}  {y:.4f}\n")
            file.write("}; \\label{plot:nonRIS_NU"+f"{NU[0]}"+"}\n")

    # aU1deraU1 = np.dot(aU1.conj(), deraU1)/(np.linalg.norm(aU1)*np.linalg.norm(deraU1))
    # nu1dernu1 = np.dot(nu1.conj(), dernu1)/(np.linalg.norm(nu1)*np.linalg.norm(dernu1))
    # aU2deraU2 = np.dot(aU2.conj(), deraU2)/(np.linalg.norm(aU2)*np.linalg.norm(deraU2))
    # nu2dernu2 = np.dot(nu2.conj(), dernu2)/(np.linalg.norm(nu2)*np.linalg.norm(dernu2))
    # print(np.abs(aU1deraU1)**2, np.abs(nu1dernu1)**2, np.abs(aU2deraU2)**2, np.abs(nu2dernu2)**2)
