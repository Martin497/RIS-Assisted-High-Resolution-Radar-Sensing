# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:00:55 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Prepare data for plots in figure 9a and 9b.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import auc


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
    load_processed_results = False
    make_plots = True
    RISth = True

    rcs_iters = 1
    fading_sims = 100
    L = 3
    RunSpectralDetection = False
    RunOMPDetection = True

    pFA_idx = 3
    th_comp = 100

    rcsplot = 0
    idx3plot = L

    noise_sims = 3

    th_spectral = 300
    confidence_level = np.logspace(-10, 0, th_spectral)
    stds = np.linspace(-1, 3.5, th_spectral)
    th_vals = np.logspace(-7, -4, th_spectral)
    th_factor = np.linspace(0, 1, th_spectral)
    residual_thN = np.linspace(2.8*1e-05, 4e-05, th_spectral)
    residual_thR = np.linspace(2.8*1e-05, 4e-05, th_spectral)

    for savename in savenames:
        print(savename)
        if load_processed_results is True:
            with open(f"{folder}/res{savename}_processed.pickle", "rb") as file:
                resDetection = pickle.load(file)

            pD_spectral_nonRIS, pD_spectral_RIS, pD_spectral, pD_spectral_joint = resDetection["pD_spectral_nonRIS"], resDetection["pD_spectral_RIS"], resDetection["pD_spectral"], resDetection["pD_spectral_joint"]
            pFA_spectral_nonRIS, pFA_spectral_RIS, pFA_spectral, pFA_spectral_joint = resDetection["pFA_spectral_nonRIS"], resDetection["pFA_spectral_RIS"], resDetection["pFA_spectral"], resDetection["pFA_spectral_joint"]
            AUC_spectral_nonRIS, AUC_spectral_RIS, AUC_spectral, AUC_spectral_joint = resDetection["AUC_spectral_nonRIS"], resDetection["AUC_spectral_RIS"], resDetection["AUC_spectral"], resDetection["AUC_spectral_joint"]
            pD_arr_nonRIS, pD_arr_RIS, pD_arr_joint = resDetection["pD_arr_nonRIS"], resDetection["pD_arr_RIS"], resDetection["pD_arr_joint"]
            AUC_nonRIS, AUC_RIS, AUC_joint = resDetection["AUC_nonRIS"], resDetection["AUC_RIS"], resDetection["AUC_joint"]
        else:
            with open(f"{folder}/res{savename}.pickle", "rb") as file:
                res = pickle.load(file)
            if RunSpectralDetection is True:
                print(f"rcs iters: {len(res)}, fading sims: {len(res['0'])}, noise sims: {len(res['0']['0']['1'])}")
            else:
                print(f"rcs iters: {len(res)}, fading sims: {len(res['0'])}")
            # =============================================================================
            # Setup
            # =============================================================================
            if RunSpectralDetection is True or RunOMPDetection is True:
                DetectionN = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, L+1, th_spectral), dtype=bool)
                DetectionR = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, L+1, th_spectral), dtype=bool)
                Detection = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, L+1, th_spectral), dtype=bool)
                Detection_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, L+1, th_spectral), dtype=bool)
                FalseAlarmN = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                FalseAlarmR = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                FalseAlarm = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                FalseAlarm_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                CorrectDetectionN = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                CorrectDetectionR = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                CorrectDetection_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral), dtype=bool)
                if RunOMPDetection is True:
                    Detection_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, L+1, th_spectral**2), dtype=bool)
                    FalseAlarm_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral**2), dtype=bool)
                    CorrectDetection_joint = np.zeros((rcs_iters, fading_sims, L+1, noise_sims, th_spectral**2), dtype=bool)

            # =============================================================================
            # Load data and compute KPIs
            # =============================================================================
            for idx1 in range(rcs_iters):
                for idx2 in range(fading_sims):
                    print(idx1, idx2)
                    output = res[f"{idx1}"][f"{idx2}"]
                    try:
                        pFA = np.insert(output["pFA"], 0, 0)
                    except:
                        th_comp = 100
                        pFA = np.insert(np.linspace(1e-09, 1, th_comp), 0, 0)
                    if RunSpectralDetection is True:
                        for idx3 in range(L+1):
                            for idx4 in range(noise_sims):
                                output = res[f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"]
                                p_valueN = output["p_valueN"]
                                p_valueR = output["p_valueR"]
                                p_value = output["p_value"]

                                if RISth is True:
                                    local_max_vals = output["local_max_val"]
                                    try:
                                        MarrR = np.sum(local_max_vals[:, None] >= np.max(local_max_vals)*th_factor[None, :], axis=0)
                                    except ValueError:
                                        MarrR = 0
                                else:
                                    MarrR = np.sum(np.array(p_valueR)[:, None] >= confidence_level[None, :], axis=0).astype(np.int8)

                                MarrN = np.sum(np.array(p_valueN)[:, None] >= confidence_level[None, :], axis=0).astype(np.int8)
                                Marr = np.sum(np.array(p_value)[:, None] >= confidence_level[None, :], axis=0).astype(np.int8)

                                for l in range(idx3+1):
                                    DetectionN[idx1, idx2, idx3, idx4, l, :] = np.where(MarrN >= l, True, False)
                                    DetectionR[idx1, idx2, idx3, idx4, l, :] = np.where(MarrR >= l, True, False)
                                    Detection[idx1, idx2, idx3, idx4, l, :] = np.where(Marr >= l, True, False)
                                    Detection_joint[idx1, idx2, idx3, idx4, l, :] = np.where(np.logical_or(MarrN >= l, MarrR >= l), True, False)

                                FalseAlarmN[idx1, idx2, idx3, idx4, :] = np.where(MarrN > idx3, True, False)
                                FalseAlarmR[idx1, idx2, idx3, idx4, :] = np.where(MarrR > idx3, True, False)
                                FalseAlarm[idx1, idx2, idx3, idx4, :] = np.where(Marr > idx3, True, False)
                                FalseAlarm_joint[idx1, idx2, idx3, idx4, :] = np.where(np.logical_or(MarrN > idx3, MarrR > idx3), True, False)

                                CorrectDetectionN[idx1, idx2, idx3, idx4, :] = np.where(MarrN == idx3, True, False)
                                CorrectDetectionR[idx1, idx2, idx3, idx4, :] = np.where(MarrR == idx3, True, False)
                                CorrectDetection_joint[idx1, idx2, idx3, idx4, :] = np.where(np.logical_or(MarrN == idx3, MarrR == idx3), True, False)
                    elif RunOMPDetection is True:
                        for idx3 in range(1, L+1):
                            for idx4 in range(noise_sims):
                                output = res[f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"]
                                errorN = np.array(output["errorN"])
                                errorR = np.array(output["errorR"])

                                MarrN = np.sum(errorN[:, None] >= residual_thN[None, :], axis=0)
                                MarrR = np.sum(errorR[:, None] >= residual_thR[None, :], axis=0)
                                Marr = np.sum(np.logical_or(errorN[:, None, None] >= residual_thN[None, :, None], errorR[:, None, None] >= residual_thR[None, None, :]), axis=0).flatten()

                                CorrectDetectionN[idx1, idx2, idx3, idx4, :] = np.where(MarrN == idx3, True, False)
                                CorrectDetectionR[idx1, idx2, idx3, idx4, :] = np.where(MarrR == idx3, True, False)
                                CorrectDetection_joint[idx1, idx2, idx3, idx4, :] = np.where(Marr == idx3, True, False)

            if RunOMPDetection is True:
                equal_to_one_threshold = 1e-02
                residual_thN_list = list()
                residual_thR_list = list()
                for idx1 in range(rcs_iters):
                    for idx3 in range(1, L+1):
                        min_thN_list = list()
                        min_thR_list = list()
                        for i in range(1, idx3+1):
                            pCDN = np.mean(CorrectDetectionN[idx1, :, i, :, :], axis=(0, 1))
                            pCDR = np.mean(CorrectDetectionR[idx1, :, i, :, :], axis=(0, 1))
                            for j in range(len(pCDN)):
                                if pCDN[j] >= 1-equal_to_one_threshold:
                                    min_thN_list.append(residual_thN[j])
                                    break
                            for j in range(len(pCDR)):
                                if pCDR[j] >= 1-equal_to_one_threshold:
                                    min_thR_list.append(residual_thR[j])
                                    break
                        residual_thN_list.append(np.linspace(max(min_thN_list), 4.5*1e-05, th_spectral))
                        residual_thR_list.append(np.linspace(max(min_thR_list), 4.5*1e-05, th_spectral))
                        for idx2 in range(fading_sims):
                            for idx4 in range(noise_sims):
                                output = res[f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"]
                                errorN = np.array(output["errorN"])
                                errorR = np.array(output["errorR"])

                                MarrN = np.sum(errorN[:, None] >= residual_thN_list[-1][None, :], axis=0)
                                MarrR = np.sum(errorR[:, None] >= residual_thR_list[-1][None, :], axis=0)
                                Marr = np.sum(np.logical_or(errorN[:, None, None] >= residual_thN_list[-1][None, :, None], errorR[:, None, None] >= residual_thR_list[-1][None, None, :]), axis=0).flatten()

                                for l in range(1, idx3+1):
                                    DetectionN[idx1, idx2, idx3, idx4, l, :] = np.where(MarrN >= l, True, False)
                                    DetectionR[idx1, idx2, idx3, idx4, l, :] = np.where(MarrR >= l, True, False)
                                    Detection_joint[idx1, idx2, idx3, idx4, l, :] = np.where(Marr >= l, True, False)

                                FalseAlarmN[idx1, idx2, idx3, idx4, :] = np.where(MarrN > idx3, True, False)
                                FalseAlarmR[idx1, idx2, idx3, idx4, :] = np.where(MarrR > idx3, True, False)
                                FalseAlarm_joint[idx1, idx2, idx3, idx4, :] = np.where(Marr > idx3, True, False)

            if RunSpectralDetection is True or RunOMPDetection is True:
                pD_spectral_nonRIS = np.mean(DetectionN, axis=3)
                pD_spectral_RIS = np.mean(DetectionR, axis=3)
                pD_spectral = np.mean(Detection, axis=3)
                pD_spectral_joint = np.mean(Detection_joint, axis=3)
        
                pFA_spectral_nonRIS = np.mean(FalseAlarmN, axis=3)
                pFA_spectral_RIS = np.mean(FalseAlarmR, axis=3)
                pFA_spectral = np.mean(FalseAlarm, axis=3)
                pFA_spectral_joint = np.mean(FalseAlarm_joint, axis=3)
        
                AUC_spectral_nonRIS = np.zeros((rcs_iters, fading_sims, L+1, L+1))
                AUC_spectral_RIS = np.zeros((rcs_iters, fading_sims, L+1, L+1))
                AUC_spectral = np.zeros((rcs_iters, fading_sims, L+1, L+1))
                AUC_spectral_joint = np.zeros((rcs_iters, fading_sims, L+1, L+1))

                pD_spectral_nonRIS_interp = np.zeros((rcs_iters, fading_sims, L+1, L+1, th_comp+2))
                pD_spectral_RIS_interp = np.zeros((rcs_iters, fading_sims, L+1, L+1, th_comp+2))
                pD_spectral_joint_interp = np.zeros((rcs_iters, fading_sims, L+1, L+1, th_comp+2))
                for idx1 in range(rcs_iters):
                    for idx2 in range(fading_sims):
                        for idx3 in range(L+1):
                            for l in range(idx3+1):
                                sort_nonRIS = np.argsort(pFA_spectral_nonRIS[idx1, idx2, l])
                                pD_temp = np.append(np.insert(pD_spectral_nonRIS[idx1, idx2, idx3, l][sort_nonRIS], 0, 0), 1)
                                pFA_temp = np.append(np.insert(pFA_spectral_nonRIS[idx1, idx2, l][sort_nonRIS], 0, 0), 1)
                                pD_temp[pFA_temp == 0] = np.sort(pD_temp[pFA_temp == 0])
                                FAnonzeros = len(pFA_temp[pFA_temp != 0])
                                pFA_temp = pFA_temp[-FAnonzeros-1:]
                                pD_temp = pD_temp[-FAnonzeros-1:]
                                pD_spectral_nonRIS_interp[idx1, idx2, idx3, l] = np.insert(np.interp(pFA, pFA_temp, pD_temp), 0, 0)

                                sort_RIS = np.argsort(pFA_spectral_RIS[idx1, idx2, l])
                                pD_temp = np.append(np.insert(pD_spectral_RIS[idx1, idx2, idx3, l][sort_RIS], 0, 0), 1)
                                pFA_temp = np.append(np.insert(pFA_spectral_RIS[idx1, idx2, l][sort_RIS], 0, 0), 1)
                                pD_temp[pFA_temp == 0] = np.sort(pD_temp[pFA_temp == 0])
                                FAnonzeros = len(pFA_temp[pFA_temp != 0])
                                pFA_temp = pFA_temp[-FAnonzeros-1:]
                                pD_temp = pD_temp[-FAnonzeros-1:]
                                pD_spectral_RIS_interp[idx1, idx2, idx3, l] = np.insert(np.interp(pFA, pFA_temp, pD_temp), 0, 0)

                                sort_joint = np.argsort(pFA_spectral_joint[idx1, idx2, l])
                                pD_temp = np.append(np.insert(pD_spectral_joint[idx1, idx2, idx3, l][sort_joint], 0, 0), 1)
                                pFA_temp = np.append(np.insert(pFA_spectral_joint[idx1, idx2, l][sort_joint], 0, 0), 1)
                                pD_temp[pFA_temp == 0] = np.sort(pD_temp[pFA_temp == 0])
                                FAnonzeros = len(pFA_temp[pFA_temp != 0])
                                pFA_temp = pFA_temp[-FAnonzeros-1:]
                                pD_temp = pD_temp[-FAnonzeros-1:]
                                pD_spectral_joint_interp[idx1, idx2, idx3, l] = np.insert(np.interp(pFA, pFA_temp, pD_temp), 0, 0)

                for idx1 in range(rcs_iters):
                    for idx2 in range(fading_sims):
                        for idx3 in range(L+1):
                            for l in range(idx3+1):
                                AUC_spectral_nonRIS[idx1, idx2, idx3, l] = auc(np.insert(pFA, 0, 0), pD_spectral_nonRIS_interp[idx1, idx2, idx3, l])
                                AUC_spectral_RIS[idx1, idx2, idx3, l] = auc(np.insert(pFA, 0, 0), pD_spectral_RIS_interp[idx1, idx2, idx3, l])
                                AUC_spectral_joint[idx1, idx2, idx3, l] = auc(np.insert(pFA, 0, 0), pD_spectral_joint_interp[idx1, idx2, idx3, l])

            resDetection = {"pD_spectral_nonRIS": pD_spectral_nonRIS, "pD_spectral_RIS": pD_spectral_RIS, "pD_spectral": pD_spectral, "pD_spectral_joint": pD_spectral_joint,
                            "pD_spectral_nonRIS_interp": pD_spectral_nonRIS_interp, "pD_spectral_RIS_interp": pD_spectral_RIS_interp, "pD_spectral_joint_interp": pD_spectral_joint_interp, "pFA": pFA,
                            "pFA_spectral_nonRIS": pFA_spectral_nonRIS, "pFA_spectral_RIS": pFA_spectral_RIS, "pFA_spectral": pFA_spectral, "pFA_spectral_joint": pFA_spectral_joint,
                            "AUC_spectral_nonRIS": AUC_spectral_nonRIS, "AUC_spectral_RIS": AUC_spectral_RIS, "AUC_spectral_joint": AUC_spectral_joint}

            with open(f"{folder}/res{savename}_processed.pickle", "wb") as file:
                pickle.dump(resDetection, file, pickle.HIGHEST_PROTOCOL)

        if make_plots is True:
            if RunSpectralDetection is True or RunOMPDetection is True:
                # =============================================================================
                # Spectral detection plots
                # =============================================================================
                if RunSpectralDetection is True:
                    ### Correct detection plots ###
                    linestyles=["solid", "dashed"]
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twiny()
                    ax1.set_xlabel("Confidence level", color="tab:blue")
                    ax1.set_ylabel("Probability of correct detection")
                    ax1.set_xscale("log")
                    ax1.tick_params(axis="x", labelcolor="tab:blue")
                    ax2.set_xlabel("Threshold factor", color="tab:orange")
                    ax2.tick_params(axis="x", labelcolor="tab:orange")
                    for l, ls in enumerate(linestyles):
                        ax1.plot(confidence_level, np.mean(CorrectDetectionN[0, :, l+1, :], axis=(0, 1)), color="tab:blue", linestyle=ls, label=f"Non-RIS, Targets: {l+1}")
                        ax2.plot(th_factor, np.mean(CorrectDetectionR[0, :, l+1, :], axis=(0, 1)), color="tab:orange", linestyle=ls, label=f"RIS, Targets: {l+1}")
                    plt.ylim(-0.03, 1.03)
                    plt.savefig(f"{folder}/PCD_Spacing{savename}.png", dpi=500, bbox_inches="tight")
                    plt.show()
                elif RunOMPDetection is True:
                    ### Correct detection plots ###
                    linestyles=["solid", "dashed", "dotted"]
                    plt.xlabel("Threshold factor")
                    for l, ls in enumerate(linestyles):
                        plt.plot(residual_thN, np.mean(CorrectDetectionN[0, :, l+1, :], axis=(0, 1)), color="tab:blue", linestyle=ls, label=f"Non-RIS, Targets: {l+1}")
                        plt.plot(residual_thR, np.mean(CorrectDetectionR[0, :, l+1, :], axis=(0, 1)), color="tab:orange", linestyle=ls, label=f"RIS, Targets: {l+1}")
                    plt.ylim(-0.03, 1.03)
                    plt.legend()
                    plt.savefig(f"{folder}/PCD_Spacing{savename}.png", dpi=500, bbox_inches="tight")
                    plt.show()

