# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:45:29 2024

@author: BX98LW
"""

import numpy as np
import matplotlib.pyplot as plt
import toml

from sklearn.metrics import auc

from TheoreticalAnalysis import TheoreticalInsights


def plot_Epd_spacing_Targetl(l):
    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, pdr_tot[idx1, idx2, :, l, 0], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    linestyle_list = ["solid", "dashed", "dotted"]
    for idx1, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
        plt.plot(pos_listN, pdn_tot[idx1, :, l, 0], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xscale("log")
    plt.grid(True, which="both")
    # plt.legend(loc="upper left")
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Expected detection probability")
    plt.xlim(10**(-3), 10**(-0.6))
    # plt.ylim(-0.02, 1.02)
    # plt.axvline(0.15, ymin=0.046, ymax=0.97, color="k")
    plt.axvline(0.15, ymin=0, ymax=1, color="k")
    plt.title(f"Target {l+1}")
    # plt.savefig(f"results/Theoretical/Epd_spacing_Target{l+1}.png", dpi=500, bbox_inches="tight")
    plt.show()

def txt_Epd_spacing_Targetl(l):
    with open(f"results/Theoretical/Epd_spacing_Target{l+1}.txt", "w") as file:
        color_list = ["color3", "color5"]
        linestyle_list = ["dotted", "dashed", "solid"]
        for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
            for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
                file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {col}, {ls}]\n")
                file.write("table {%\n")
                for x, y in zip(pos_listR, pdr_tot[idx1, idx2, :, l, 0]):
                    file.write(f"{x:.15f}  {y:.15f}\n")
                file.write("}; \\label{plot:Epd_RIS_T"+f"{T2//2}_NR{NR[0]}"+"}\n")
        for idx1, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
            file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {ls}]\n")
            file.write("table {%\n")
            for x, y in zip(pos_listN, pdn_tot[idx1, :, l, 0]):
                file.write(f"{x:.15f}  {y:.15f}\n")
            file.write("}; \\label{plot:Epd_nonRIS_NU"+f"{NU[0]}"+"}\n")

def plot_AUC_Epd_spacing_Targetl(l):
    color_list = ["firebrick", "darkorange"]
    linestyle_list = ["dotted", "dashed", "solid"]
    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
        for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
            plt.plot(pos_listR, AUCr[idx1, idx2, :, l], color=col, linestyle=ls, label=f"RIS: $N_r$ = {NR[0]} x {NR[1]}; T = {T2}")
    linestyle_list = ["solid", "dashed", "dotted"]
    for idx1, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
        plt.plot(pos_listN, AUCn[idx1, :, l], color="midnightblue", linestyle=ls, label=f"Non-RIS: $N_u$ = {NU[0]} x {NU[1]}")
    plt.xscale("log")
    plt.grid(True, which="both")
    # plt.legend(loc="upper left")
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("Area under the ROC curve (AUC)")
    plt.xlim(10**(-3), 10**(-0.6))
    # plt.ylim(-0.02, 1.02)
    # plt.axvline(0.15, ymin=0.046, ymax=0.97, color="k")
    plt.axvline(0.15, ymin=0, ymax=1, color="k")
    plt.title(f"Target {l+1}")
    # plt.savefig(f"results/Theoretical/AUC_Epd_spacing_Target{l+1}.png", dpi=500, bbox_inches="tight")
    plt.show()

def txt_AUC_Epd_spacing_Targetl(l):
    with open(f"results/Theoretical/AUCEpd_spacing_Target{l+1}.txt", "w") as file:
        color_list = ["color3", "color5"]
        linestyle_list = ["dotted", "dashed", "solid"]
        for idx1, (col, NR) in enumerate(zip(color_list, NR_list)):
            for idx2, (ls, T2) in enumerate(zip(linestyle_list, T2_list)):
                file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {col}, {ls}]\n")
                file.write("table {%\n")
                for x, y in zip(pos_listR, AUCr[idx1, idx2, :, l]):
                    file.write(f"{x:.15f}  {y:.15f}\n")
                file.write("}; \\label{plot:AUC_Epd_RIS_T"+f"{T2//2}_NR{NR[0]}"+"}\n")
        for idx1, (ls, NU) in enumerate(zip(linestyle_list, NU_list)):
            file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {ls}]\n")
            file.write("table {%\n")
            for x, y in zip(pos_listN, AUCn[idx1, :, l]):
                file.write(f"{x:.15f}  {y:.15f}\n")
            file.write("}; \\label{plot:AUC_Epd_nonRIS_NU"+f"{NU[0]}"+"}\n")


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
    # Test expected detection probability
    # =============================================================================
    th_comp = 100
    pFA_arr = np.logspace(-7, 0, th_comp)
    # pFA_arr[-1] = 0.999
    # pFA_arr = np.hstack((pFA_arr, 0.99, 0.9999, 0.9999999))

    mod = TheoreticalInsights(None, **toml_settings)
    rcs = np.sqrt(np.array([50, 5, 0.5]))
    Phi_taus = np.array([60, 60, 60])
    az0 = 0.7
    el0 = 0.8
    Delta = 0.1
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array([az0, az0+Delta, az0-Delta])
    Phi_els = np.array([el0, el0-Delta, el0+Delta])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

    pd1n, pd2n, pd3n, pd1r, pd2r, pd3r = mod.ComputeExpectedDetectionProbability(pFA_arr, Phi, sU, rcs, toml_estimation, False, bounds)

    pD1joint = pd1n + pd1r - pd1n*pd1r
    pD2joint = pd2n + pd2r - pd2n*pd2r
    pD3joint = pd3n + pd3r - pd3n*pd3r
    pFAjoint = 2*pFA_arr - pFA_arr**2

    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    ax = fig.add_subplot(111)
    plt.plot(pFA_arr, pd1r, color="tab:orange", linestyle="solid", label="RIS: Target 1")
    plt.plot(pFA_arr, pd1n, color="midnightblue", linestyle="solid", label="Non-RIS: Target 1")
    plt.plot(pFAjoint, pD1joint, color="tab:purple", linestyle="solid", label="Joint: Target 1")
    plt.plot(pFA_arr, pd2r, color="tab:orange", linestyle="dashed", label="RIS: Target 2")
    plt.plot(pFA_arr, pd2n, color="midnightblue", linestyle="dashed", label="Non-RIS: Target 2")
    plt.plot(pFAjoint, pD2joint, color="tab:purple", linestyle="dashed", label="Joint: Target 2")
    plt.plot(pFA_arr, pd3r, color="tab:orange", linestyle="dotted", label="RIS: Target 3")
    plt.plot(pFA_arr, pd3n, color="midnightblue", linestyle="dotted", label="Non-RIS: Target 3")
    plt.plot(pFAjoint, pD3joint, color="tab:purple", linestyle="dotted", label="Joint: Target 3")
    plt.xlabel("False alarm probability")
    plt.ylabel("Expected detection probability")
    # plt.grid(True, which="both")
    # plt.xscale("log")
    plt.legend()
    # plt.savefig("results/Theoretical/Epd_ROC.png", dpi=500, bbox_inches="tight")
    plt.show()

    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    ax = fig.add_subplot(111)
    plt.plot(pFA_arr, 1-pd1r, color="tab:orange", linestyle="solid", label="RIS: Target 1")
    plt.plot(pFA_arr, 1-pd1n, color="midnightblue", linestyle="solid", label="Non-RIS: Target 1")
    plt.plot(pFAjoint, 1-pD1joint, color="tab:purple", linestyle="solid", label="Joint: Target 1")
    plt.plot(pFA_arr, 1-pd2r, color="tab:orange", linestyle="dashed", label="RIS: Target 2")
    plt.plot(pFA_arr, 1-pd2n, color="midnightblue", linestyle="dashed", label="Non-RIS: Target 2")
    plt.plot(pFAjoint, 1-pD2joint, color="tab:purple", linestyle="dashed", label="Joint: Target 2")
    plt.plot(pFA_arr, 1-pd3r, color="tab:orange", linestyle="dotted", label="RIS: Target 3")
    plt.plot(pFA_arr, 1-pd3n, color="midnightblue", linestyle="dotted", label="Non-RIS: Target 3")
    plt.plot(pFAjoint, 1-pD3joint, color="tab:purple", linestyle="dotted", label="Joint: Target 3")
    plt.xlabel("False alarm probability")
    plt.ylabel("Expected misdetection probability")
    plt.grid(True, which="both")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min(pFAjoint), 1.03)
    plt.legend(loc="lower left")
    # plt.savefig("results/Theoretical/Epmd_ROC.png", dpi=500, bbox_inches="tight")
    plt.show()

    with open("results/Theoretical/Epmd_ROC.txt", "w") as file:
        file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {'solid'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd1n):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:nonRIS_Target1}\n")
        file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {'dashed'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd2n):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:nonRIS_Target2}\n")
        file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {'dotted'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd3n):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:nonRIS_Target3}\n")

        file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {'color3'}, {'solid'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd1r):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:RIS_Target1}\n")
        file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {'color3'}, {'dashed'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd2r):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:RIS_Target2}\n")
        file.write(f"\\addplot [semithick, mark=square, mark options={'solid'}, {'color3'}, {'dotted'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFA_arr, 1-pd3r):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:RIS_Target3}\n")

        file.write(f"\\addplot [semithick, mark=diamond, mark options={'solid'}, {'color4'}, {'solid'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFAjoint, 1-pD1joint):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:joint_Target1}\n")
        file.write(f"\\addplot [semithick, mark=diamond, mark options={'solid'}, {'color4'}, {'dashed'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFAjoint, 1-pD2joint):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:joint_Target2}\n")
        file.write(f"\\addplot [semithick, mark=diamond, mark options={'solid'}, {'color4'}, {'dotted'}]\n")
        file.write("table {%\n")
        for x, y in zip(pFAjoint, 1-pD3joint):
            file.write(f"{x:.15f}  {y:.15f}\n")
        file.write("}; \\label{plot:joint_Target3}\n")

    # =============================================================================
    # Expected detection probability wrt. Delta
    # =============================================================================
    pos_listN = np.logspace(-3, -0.6, 60)
    pos_listR = np.logspace(-3, -0.6, 60)
    MR = len(pos_listR)
    MN = len(pos_listN)
    th_comp = 100
    pFA_arr = np.linspace(1e-09, 1, th_comp)

    NR_list = [[15, 15], [35, 35]]
    T2_list = [8, 18, 50]
    pdr_tot = np.zeros((len(NR_list), len(T2_list), MR, 3, th_comp))
    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            toml_settings["NR"] = NR
            toml_settings["T2"] = T2
            mod = TheoreticalInsights(None, **toml_settings)
            for pos_idx, Delta in enumerate(pos_listR):
                rcs = np.sqrt(np.array([50, 5, 0.5]))
                Phi_taus = np.array([60, 60, 60])
                az0 = 0.7
                el0 = 0.8
                Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
                Phi_azs = np.array([az0, az0+Delta, az0-Delta])
                Phi_els = np.array([el0, el0-Delta, el0+Delta])
                Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

                _, _, _, pd1r, pd2r, pd3r = mod.ComputeExpectedDetectionProbability(pFA_arr, Phi, sU, rcs, toml_estimation, False, bounds)
                pdr_tot[idx1, idx2, pos_idx, 0] = pd1r
                pdr_tot[idx1, idx2, pos_idx, 1] = pd2r
                pdr_tot[idx1, idx2, pos_idx, 2] = pd3r

    NU_list = [[2, 2], [4, 4], [8, 8]]
    pdn_tot = np.zeros((len(NU_list), MN, 3, th_comp))
    for idx1, NU in enumerate(NU_list):
        toml_settings["NU"] = NU
        mod = TheoreticalInsights(None, **toml_settings)
        for pos_idx, Delta in enumerate(pos_listN):
            rcs = np.sqrt(np.array([50, 5, 0.5]))
            Phi_taus = np.array([60, 60, 60])
            az0 = 0.7
            el0 = 0.8
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array([az0, az0+Delta, az0-Delta])
            Phi_els = np.array([el0, el0-Delta, el0+Delta])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
            pd1n, pd2n, pd3n, _, _, _ = mod.ComputeExpectedDetectionProbability(pFA_arr, Phi, sU, rcs, toml_estimation, False, bounds)
            pdn_tot[idx1, pos_idx, 0] = pd1n
            pdn_tot[idx1, pos_idx, 1] = pd2n
            pdn_tot[idx1, pos_idx, 2] = pd3n

    plot_Epd_spacing_Targetl(0)
    plot_Epd_spacing_Targetl(1)
    plot_Epd_spacing_Targetl(2)

    txt_Epd_spacing_Targetl(0)
    txt_Epd_spacing_Targetl(1)
    txt_Epd_spacing_Targetl(2)

    pDjoint = pdn_tot[None, None, :, :, :] + pdr_tot[:, :, None, :, :] - pdn_tot[None, None, :, :, :]*pdr_tot[:, :, None, :, :]
    pFAjoint = 2*pFA_arr - pFA_arr**2
    AUCjoint = np.zeros((len(NR_list), len(T2_list), len(NU_list), MR, 3))
    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            for idx3, NU in enumerate(NU_list):
                for pos_idx, Delta in enumerate(pos_listR):
                    AUCjoint[idx1, idx2, idx3, pos_idx, 0] = auc(np.insert(pFAjoint, 0, 0), np.insert(pDjoint[idx1, idx2, idx3, pos_idx, 0], 0, 0))
                    AUCjoint[idx1, idx2, idx3, pos_idx, 1] = auc(np.insert(pFAjoint, 0, 0), np.insert(pDjoint[idx1, idx2, idx3, pos_idx, 1], 0, 0))
                    AUCjoint[idx1, idx2, idx3, pos_idx, 2] = auc(np.insert(pFAjoint, 0, 0), np.insert(pDjoint[idx1, idx2, idx3, pos_idx, 2], 0, 0))

    for l in range(1, 3):
        with open(f"results/Theoretical/AUC_joint_spacing_target{l+1}.txt", "w") as file:
            file.write(f"\\addplot [semithick, mark=star, mark options={'solid'}, {'color2'}, {'solid'}]\n")
            file.write("table {%\n")
            for x, y in zip(pos_listR, AUCjoint[-1, -1, 0, l]):
                file.write(f"{x:.15f}  {y:.15f}\n")
            file.write("};\n")

    # =============================================================================
    # AUC of expected detection proability wrt. Delta
    # =============================================================================
    AUCr = np.zeros((len(NR_list), len(T2_list), MR, 3))
    AUCn = np.zeros((len(NU_list), MR, 3))

    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            for pos_idx, Delta in enumerate(pos_listR):
                AUCr[idx1, idx2, pos_idx, 0] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdr_tot[idx1, idx2, pos_idx, 0], 0, 0))
                AUCr[idx1, idx2, pos_idx, 1] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdr_tot[idx1, idx2, pos_idx, 1], 0, 0))
                AUCr[idx1, idx2, pos_idx, 2] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdr_tot[idx1, idx2, pos_idx, 2], 0, 0))

    for idx1, NU in enumerate(NU_list):
        for pos_idx, Delta in enumerate(pos_listN):
            AUCn[idx1, pos_idx, 0] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdn_tot[idx1, pos_idx, 0], 0, 0))
            AUCn[idx1, pos_idx, 1] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdn_tot[idx1, pos_idx, 1], 0, 0))
            AUCn[idx1, pos_idx, 2] = auc(np.insert(pFA_arr, 0, 0), np.insert(pdn_tot[idx1, pos_idx, 2], 0, 0))

    plot_AUC_Epd_spacing_Targetl(0)
    plot_AUC_Epd_spacing_Targetl(1)
    plot_AUC_Epd_spacing_Targetl(2)

    txt_AUC_Epd_spacing_Targetl(0)
    txt_AUC_Epd_spacing_Targetl(1)
    txt_AUC_Epd_spacing_Targetl(2)

    # =============================================================================
    # Joint detection
    # =============================================================================
    # for idx1, NR in enumerate(NR_list):
    #     for idx2, T2 in enumerate(T2_list):
    #         for idx3, NU in enumerate(NU_list):
    #             plt.plot(AUCjoint[idx1, idx2, idx3, :, 2], color="tab:purple")
    #             plt.plot(AUCr[idx1, idx2, :, 2], color="tab:orange")
    #             plt.plot(AUCn[idx3, :, 2], color="tab:blue")
    #             plt.show()

    # =============================================================================
    # Detection probability lower bound
    # =============================================================================
    # pos_listR = np.linspace(0.04, 0.15, 100)
    # NR_list = [[35, 35], [50, 50]]
    # T2_list = [18, 50]
    # pdr_lower_tot = np.zeros((len(NR_list), len(T2_list), len(pos_listR)))
    # for idx1, NR in enumerate(NR_list):
    #     for idx2, T2 in enumerate(T2_list):
    #         toml_settings["NR"] = NR
    #         toml_settings["T2"] = T2
    #         mod = TheoreticalInsights(None, **toml_settings)
    #         for pos_idx, Delta in enumerate(pos_listR):
    #             rcs = np.sqrt(np.array([50, 50]))
    #             Phi_taus = np.array([60, 60])
    #             az0 = 0.7
    #             el0 = 0.8
    #             Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    #             Phi_azs = np.array([az0, az0+Delta])
    #             Phi_els = np.array([el0, el0-Delta])
    #             Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

    #             _, pdr_lower = mod.ComputeDetectionProbabilityLowerBound(Phi, sU, rcs, toml_estimation, False, bounds)
    #             pdr_lower_tot[idx1, idx2, pos_idx] = pdr_lower

    # plt.plot(pos_listR, pdr_lower_tot[-1, -1])
    # plt.xscale("log")
    # plt.grid(True, which="both")
    # plt.xlim(7.5e-03, 10**(-0.6))
    # plt.show()