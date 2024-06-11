# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:20:03 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - 
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pickle
import toml

import multiprocessing as mp
from tqdm import tqdm

from MainEstimation import ChannelEstimation

def fun_mp(i, seed, Phi, ChPars, rcs, prior, sU, toml_estimation, **kwargs):
    """
    Run scenario and get sensing algorithm results.
    """
    np.random.seed(seed+i)
    mod = ChannelEstimation(None, False, False, **kwargs)
    resDict = mod(Phi, sU, rcs, toml_estimation, ChPars=ChPars, prior=prior)
    return resDict

def sim_mp(sims, N_workers, *args, **kwargs):
    """
    Multiprocessing: Parallelisation over Monte Carlo runs.
    """
    # Pool with progress bar
    pool = mp.Pool(processes=N_workers,
                   initargs=(mp.RLock(),),
                   initializer=tqdm.set_lock)

    # run multiprocessing
    jobs = [pool.apply_async(fun_mp, args=(i, *args), kwds=kwargs) for i in range(sims)]
    pool.close()
    pool.join()

    # stack results
    res_dict = dict()
    for i in range(sims):
        res_dict[f"{i}"] = jobs[i].get()
    return res_dict


def main():
    # =============================================================================
    # Load configuration file
    # =============================================================================
    config_file = "system_config.toml"
    toml_in = toml.load(config_file)
    toml_settings = toml_in["settings"]
    toml_positions = toml_in["positions"]
    toml_estimation = toml_in["estimation"]

    # =============================================================================
    # Implement setting dictionaries
    # =============================================================================
    seed = toml_settings["seed"]
    np.random.seed(seed)
    mod = ChannelEstimation(None, False, False, **toml_settings)
    sU = np.array(toml_settings["sU"])

    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    # =============================================================================
    # Run simulation study
    # =============================================================================
    savename = "PosOMP06"
    folder = f"results/Main/{savename}"
    use_mp = False

    rcs_iters = 1
    fading_sims = 15
    noise_sims = 1

    with open(f"{folder}/{savename}.toml", "w") as toml_file:
        toml.dump(toml_in, toml_file)

    res = dict()
    del toml_settings["sU"]
    del toml_settings["seed"]

    pos_list = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
    with open(f"{folder}/AOAspacings_data.txt", "w") as file:
        for pos in pos_list:
            file.write(f"{pos},")
    pos_iters = len(pos_list)

    # =============================================================================
    # Run
    # =============================================================================
    rcs_setting = np.sqrt(np.array(toml_positions["Phi_rcs"]))
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
                outputDetect, outputFisher = mod.RunChAnalysis(Phi, sU, rcs_iter, toml_estimation, ChPars=ChPars, prior=prior, run_fisher=True, bounds=bounds) # Compute theoretical detection probability
                res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"] = {**outputDetect, **outputFisher, **{"rcs": rcs_iter}}

                for idx3 in range(1, L+1):
                    print(pos_idx, idx1, idx2, idx3)
                    res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"][f"{idx3}"] = dict()
                    Phi_small = Phi[:idx3]
                    rcs_small = rcs_iter[:idx3]
                    print(rcs_small)
                    ChPars_small = ChPars[:idx3]

                    if use_mp is True:
                        args = (seed, Phi_small, ChPars_small, rcs_small, prior, sU, toml_estimation)
                        N_workers = min(mp.cpu_count(), noise_sims, 12)
                        output = sim_mp(noise_sims, N_workers, *args, **toml_settings)
                        res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"][f"{idx3}"] = output
                    elif use_mp is False:
                        for idx4 in range(noise_sims):
                            print(pos_idx, idx1, idx2, idx3, idx4)
                            outputSens = mod.HiResSens(Phi_small, sU, rcs_small, toml_estimation, ChPars=ChPars_small, prior=prior, bounds=bounds)
                            res[f"{pos_idx}"][f"{idx1}"][f"{idx2}"][f"{idx3}"][f"{idx4}"] = outputSens
                            # print(f"True number: {idx3}, Est number: {outputSpectral['Marr'][50]}, {outputSpectral['MarrN'][50]}, {outputSpectral['MarrR'][50]}")

                with open(f"{folder}/{savename}_temp.pickle", "wb") as file:
                    pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

    with open(f"{folder}/{savename}.pickle", "wb") as file:
        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

