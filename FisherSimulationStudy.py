# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:29:28 2024

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

    savename = "004"
    folder = f"results/Fisher/{savename}"

    pos_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    pos_iters = len(pos_list)
    rcs_iters = 1
    fading_sims = 1000

    # bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
    #           "theta_bounds": np.array([0.57, 0.87, 0.65, 0.95]),
    #           "tau_bar_bounds": np.array([1.13e-07, 1.32e-07]),
    #           "phi_bounds": np.array([0.10, 0.78, 0.55, 1.16])}
    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    with open(f"{folder}/Fisher{savename}.toml", "w") as toml_file:
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
        else:
            Phi_taus = np.array(toml_positions["Phi_taus"])
            Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            Phi_azs = np.array(toml_positions["Phi_azs"])
            Phi_els = np.array(toml_positions["Phi_els"])
            Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

        for idx1 in range(rcs_iters):
            if idx1 > 0:
                rcs_setting[1:] *= np.sqrt(2)
            res[f"{pos_idx}"][f"{idx1}"] = dict()
            for idx2 in range(fading_sims):
                print(idx1, idx2)
                _, ChPars, rcs_iter, prior = mod.InitializeSetup(Phi, sU, rcs_setting, toml_estimation) # Simulate radar cross section
                _, outputFisher = mod.RunChAnalysis(Phi, sU, rcs_iter, toml_estimation, ChPars=ChPars, prior=prior, bounds=bounds, run_fisher=True, run_detection=False) # Compute Cramer-Rao lower bounds
                res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"] = {**outputFisher , **{"rcs": rcs_iter}}
    
                counter += 1
                if counter % 10 == 0:
                    with open(f"{folder}/{savename}_temp.pickle", "wb") as file:
                        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)
    
    with open(f"{folder}/FisherResults{savename}.pickle", "wb") as file:
        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
