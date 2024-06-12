# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:50:12 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Create the plots figure 3.
"""

import numpy as np
import toml

from MainEstimation import ChannelEstimation


def main():
    np.set_printoptions(precision=5, linewidth=np.inf)
    # =============================================================================
    # Load configuration file
    # =============================================================================
    config_file = "system_config.toml"
    toml_in = toml.load(config_file)
    toml_settings = toml_in["settings"]
    toml_estimation = toml_in["estimation"]

    # =============================================================================
    # Implement setting dictionaries
    # =============================================================================
    np.random.seed(toml_settings["seed"])
    mod = ChannelEstimation(None, True, False, **toml_settings)
    sU = np.array(toml_settings["sU"])
    toml_estimation["sparsity"] = 2
    toml_estimation["residual_thresholdN"] = 2e-05
    toml_estimation["residual_thresholdR"] = 2e-05

    # =============================================================================
    # Positions
    # =============================================================================
    Phi_taus = np.array([60, 60])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array([0.7, 0.8])
    Phi_els = np.array([0.8, 0.7])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    rcs = np.sqrt(np.array([50, 5.0]))

    # =============================================================================
    # Run algorithm
    # =============================================================================
    _ = mod.HiResSens(Phi, sU, rcs, toml_estimation, run_fisher=False, run_detection=False)


if __name__ == "__main__":
    main()