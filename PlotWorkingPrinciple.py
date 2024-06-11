# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:50:12 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - 
"""

import numpy as np
import toml
import matplotlib.pyplot as plt
import tikzplotlib

from MainEstimation import ChannelEstimation


def main():
    np.set_printoptions(precision=5, linewidth=np.inf)
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
    np.random.seed(toml_settings["seed"])
    sU = np.array(toml_settings["sU"])

    # =============================================================================
    # Read positions
    # =============================================================================
    Phi_taus = np.array(toml_positions["Phi_taus"])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array(toml_positions["Phi_azs"])
    Phi_els = np.array(toml_positions["Phi_els"])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    print(Phi)

    # =============================================================================
    # Run algorithm
    # =============================================================================
    toml_estimation["simPhi"] = "None"
    bounds = {"tau_bounds": [119*1e-09, 122*1e-09], "theta_bounds": [0.56, 0.81, 0.74, 0.88],
              "tau_bar_bounds": [121.2*1e-09, 122.5*1e-09], "phi_bounds": [0.35, 0.60, 0.82, 0.96]}
    mod = ChannelEstimation(None, True, bounds, **toml_settings)
    _ = mod.HiResSens(Phi, sU, toml_estimation)


if __name__ == "__main__":
    main()