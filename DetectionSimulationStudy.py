# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:52:55 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - 
"""


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import toml
import pickle

from MainEstimation import ChannelEstimation


def main():
    np.set_printoptions(precision=5, linewidth=np.inf)
    plt.style.use("seaborn-v0_8-darkgrid")
    # =============================================================================
    # Load configuration file
    # =============================================================================
    config_file = "system_config.toml"
    toml_in= toml.load(config_file)
    toml_settings = toml_in["settings"]
    toml_positions = toml_in["positions"]
    toml_estimation = toml_in["estimation"]

    # =============================================================================
    # Implement setting dictionaries
    # =============================================================================
    np.random.seed(toml_settings["seed"])
    mod = ChannelEstimation(None, False, False, **toml_settings)
    sU = np.array(toml_settings["sU"])

    # =============================================================================
    # Run simulation study
    # =============================================================================
    np.random.seed(toml_settings["seed"])

    savename = "TOMP03"
    folder = f"results/Detection/{savename}"
    RunSpectralDetection = True

    pos_list = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
    with open(f"{folder}/AOAspacings_data.txt", "w") as file:
        for pos in pos_list:
            file.write(f"{pos},")
    pos_iters = len(pos_list)
    rcs_iters = 1
    fading_sims = 100
    noise_sims = 3

    # bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
    #           "theta_bounds": np.array([0.57, 0.87, 0.65, 0.95]),
    #           "tau_bar_bounds": np.array([1.13e-07, 1.32e-07]),
    #           "phi_bounds": np.array([0.10, 0.78, 0.55, 1.16])}
    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    with open(f"{folder}/Detection{savename}.toml", "w") as toml_file:
        toml.dump(toml_in, toml_file)

    # =============================================================================
    # Run
    # =============================================================================
    rcs_setting = np.sqrt(np.array(toml_positions["Phi_rcs"]))
    counter = 0
    res = dict()
    for pos_idx, Delta in enumerate(pos_list):
        res[f"{pos_idx}"] = dict()
        if pos_iters > 1:
            Phi_taus = np.array([60, 60, 60])
            az0 = 0.7
            el0 = 0.8
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array([az0, az0+Delta, az0-Delta])
            Phi_els = np.array([el0, el0-Delta, el0+Delta])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
            # Phi_taus = np.array([60, 60])
            # az0 = 0.7
            # el0 = 0.8
            # Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            # Phi_azs = np.array([az0, az0+Delta])
            # Phi_els = np.array([el0, el0-Delta])
            # Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
        else:
            Phi_taus = np.array(toml_positions["Phi_taus"])
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array(toml_positions["Phi_azs"])
            Phi_els = np.array(toml_positions["Phi_els"])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
        L = Phi.shape[0]

        for idx1 in range(rcs_iters):
            if idx1 > 0:
                rcs_setting[1:] *= np.sqrt(2)
            res[f"{pos_idx}"][f"{idx1}"] = dict()
            for idx2 in range(fading_sims):
                print(pos_idx, idx1, idx2)
                _, ChPars, rcs_iter, prior = mod.InitializeSetup(Phi, sU, rcs_setting, toml_estimation) # Simulate radar cross section
                # outputDetect, _ = mod.RunChAnalysis(Phi, sU, rcs_iter, toml_estimation, ChPars=ChPars, prior=prior, bounds=bounds) # Compute theoretical detection probability
                # res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"] = {**outputDetect , **{"rcs": rcs_iter}}
                res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"] = {**{"rcs": rcs_iter}}

                if RunSpectralDetection is True:
                    for idx3 in range(1, L+1):
                        res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"][f"{idx3}"] = dict()
                        Phi_small = Phi[:idx3]
                        rcs_small = rcs_iter[:idx3]
                        print(rcs_small)
                        ChPars_small = ChPars[:idx3]
                        for idx4 in range(noise_sims):
                            print(pos_idx, idx1, idx2, idx3, idx4)
                            outputSpectral = mod.ThresholdDetection(Phi_small, sU, rcs_small, toml_estimation, ChPars=ChPars_small, prior=prior, bounds=bounds)
                            res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"] = outputSpectral
                            # print(f"True number: {idx3}, Est number: {outputSpectral['Marr'][50]}, {outputSpectral['MarrN'][50]}, {outputSpectral['MarrR'][50]}")

                counter += 1
                if counter % 10 == 0:
                    with open(f"{folder}/{savename}_temp.pickle", "wb") as file:
                        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)
    
            # L = output["pD_arr_nonRIS"].shape[0]
            # for l in range(L):
            #     AUC_nonRIS = auc(output["pFA"], output["pD_arr_nonRIS"][l])
            #     AUC_RIS = auc(output["pFA"], output["pD_arr_RIS"][l])
            #     AUC_joint = auc(2*output["pFA"]-output["pFA"]**2, output["pD_arr_nonRIS"][l]+output["pD_arr_RIS"][l]-output["pD_arr_nonRIS"][l]*output["pD_arr_RIS"][l])
            #     print(f"Area under the ROC curve, Target {l+1}:")
            #     print(f"Non-RIS : {AUC_nonRIS:.4f}\nRIS     : {AUC_RIS:.4f}\nJoint   : {AUC_joint:.4f}")

    with open(f"{folder}/DetectionResults{savename}.pickle", "wb") as file:
        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
