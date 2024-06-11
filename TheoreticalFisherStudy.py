# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:10:31 2024

@author: BX98LW
"""


import numpy as np
import matplotlib.pyplot as plt
import toml

from TheoreticalAnalysis import TheoreticalInsights


def fisher_theta(alpha1, alpha2, term1, term2, term3, term4, term5):
    information = np.abs(alpha1)**2 * term1
    il_num = np.real(np.conj(alpha1)*alpha2)**2 * term3 + np.imag(np.conj(alpha1)*alpha2)**2 * term4 - 2*np.real(np.conj(alpha1)*alpha2)*np.imag(np.conj(alpha1)*alpha2) * term5
    il_denum = np.abs(alpha2)**2 * term2
    # return np.where(information - il_num/il_denum >= 0, information - il_num/il_denum, 0)
    return information - il_num/il_denum

def functional_boxplot(x, y1):
    """
    """
    y1_05quantile = np.nanquantile(y1, q=0.05, axis=0)
    y1_25quantile = np.nanquantile(y1, q=0.25, axis=0)
    y1_50quantile = np.nanquantile(y1, q=0.50, axis=0)
    y1_75quantile = np.nanquantile(y1, q=0.75, axis=0)
    y1_95quantile = np.nanquantile(y1, q=0.95, axis=0)

    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    ax = fig.add_subplot(111)

    plt.plot(x, y1_50quantile, color="navy", label="")
    ax.fill_between(x, y1_05quantile, y1_25quantile, color="cornflowerblue")
    ax.fill_between(x, y1_25quantile, y1_50quantile, color="tab:blue")
    ax.fill_between(x, y1_50quantile, y1_75quantile, color="tab:blue")
    ax.fill_between(x, y1_75quantile, y1_95quantile, color="cornflowerblue")
    plt.plot(x, np.nanmean(y1, axis=0), color="navy", linestyle="dashed")
    
    # plt.xlabel("Normalized coherence")
    plt.ylabel("AOA Fisher information")
    plt.show()


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
    pos_list = np.logspace(-2, np.log10(0.15), 100)
    M = len(pos_list)

    NR_list = [[35, 35]]
    T2_list = [18]
    term1N_tot = np.zeros(M)
    term2N_tot = np.zeros(M)
    term3N_tot = np.zeros(M)
    term4N_tot = np.zeros(M)
    term5N_tot = np.zeros(M)
    muCN_tot = np.zeros(M)
    # EFIazN_tot = np.zeros((1000, M))
    term1R_tot = np.zeros(M)
    term2R_tot = np.zeros(M)
    term3R_tot = np.zeros(M)
    term4R_tot = np.zeros(M)
    term5R_tot = np.zeros(M)
    muCR_tot = np.zeros(M)
    # EFIazR_tot = np.zeros((1000, M))
    for idx1, NR in enumerate(NR_list):
        for idx2, T2 in enumerate(T2_list):
            toml_settings["NR"] = NR
            toml_settings["T2"] = T2
            toml_settings["NU"] = [4, 2]
            mod = TheoreticalInsights(None, **toml_settings)
            for pos_idx, Delta in enumerate(pos_list):
                rcs = np.sqrt(np.array([50, 5]))
                Phi_taus = np.array([60, 60])
                az0 = 0.7
                el0 = 0.8
                Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
                Phi_azs = np.array([az0, az0+Delta])
                Phi_els = np.array([el0, el0])
                Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

                # EFIMthetaAz1 = mod.ComputeFisherPartial(Phi, sU, rcs, toml_estimation, False, bounds)

                term1N, term2N, term3N, term4N, term5N, muCN, term1R, term2R, term3R, term4R, term5R, muCR \
                    = mod.ComputeFisherInnerProducts(Phi, sU, rcs, toml_estimation, False, bounds)
                term1N_tot[pos_idx] = term1N
                term2N_tot[pos_idx] = term2N
                term3N_tot[pos_idx] = term3N
                term4N_tot[pos_idx] = term4N
                term5N_tot[pos_idx] = term5N
                muCN_tot[pos_idx] = muCN
                # EFIazN_tot[:, pos_idx] = EFIazN

            # for pos_idx, Delta in enumerate(pos_listR):
            #     rcs = np.sqrt(np.array([50, 5]))
            #     Phi_taus = np.array([60, 60])
            #     az0 = 0.7
            #     el0 = 0.8
            #     Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
            #     Phi_azs = np.array([az0, az0+Delta])
            #     Phi_els = np.array([el0, el0-Delta])
            #     Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

            #     _, _, _, _, _, _, _, term1R, term2R, term3R, term4R, term5R, muCR, EFIazR \
            #         = mod.ComputeFisherInnerProducts(Phi, sU, rcs, toml_estimation, False, bounds)
                term1R_tot[pos_idx] = term1R
                term2R_tot[pos_idx] = term2R
                term3R_tot[pos_idx] = term3R
                term4R_tot[pos_idx] = term4R
                term5R_tot[pos_idx] = term5R
                muCR_tot[pos_idx] = muCR
                # EFIazR_tot[:, pos_idx] = EFIazR

    # plt.plot(pos_list, term1_tot, color="tab:blue")
    # plt.show()
    # plt.plot(pos_list, term2_tot, color="tab:red")
    # plt.show()
    # plt.plot(pos_list, term3_tot, color="tab:green")
    # plt.show()
    # plt.plot(pos_list, term4_tot, color="tab:orange")
    # plt.show()
    # plt.plot(pos_list, term5_tot, color="tab:olive")
    # plt.show()

    alpha_sims = 10000
    Palphal_bar = mod.Palphal_bar
    Palphal = mod.Palphal
    FIthetaN = np.zeros((alpha_sims, M))
    FIthetaR = np.zeros((alpha_sims, M))
    for i in range(alpha_sims):
        alpha1 = np.random.normal(0, np.sqrt(2/np.pi)*rcs[0]*Palphal[0]) + 1j * np.random.normal(0, np.sqrt(2/np.pi)*rcs[0]*Palphal[0])
        alpha2 = np.random.normal(0, np.sqrt(2/np.pi)*rcs[1]*Palphal[1]) + 1j * np.random.normal(0, np.sqrt(2/np.pi)*rcs[1]*Palphal[1])
        FIthetaN[i] = 4/mod.p_noise * fisher_theta(alpha1, alpha2, term1N_tot, term2N_tot, term3N_tot, term4N_tot, term5N_tot)
        alpha_bar1 = np.random.normal(0, np.sqrt(2/np.pi)*rcs[0]*Palphal_bar[0]) + 1j * np.random.normal(0, np.sqrt(2/np.pi)*rcs[0]*Palphal_bar[0])
        alpha_bar2 = np.random.normal(0, np.sqrt(2/np.pi)*rcs[1]*Palphal_bar[1]) + 1j * np.random.normal(0, np.sqrt(2/np.pi)*rcs[1]*Palphal_bar[1])
        FIthetaR[i] = 4/mod.p_noise * fisher_theta(alpha_bar1, alpha_bar2, term1R_tot, term2R_tot, term3R_tot, term4R_tot, term5R_tot)

    # functional_boxplot(muCN_tot, FIthetaN)
    # functional_boxplot(muCR_tot, FIthetaR)

    # functional_boxplot(pos_listN, FIthetaN)
    # functional_boxplot(pos_listR, FIthetaR)

    fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    ax = fig.add_subplot(111)
    plt.plot(pos_list, np.mean(FIthetaN, axis=0), color="navy")
    plt.plot(pos_list, np.mean(FIthetaR, axis=0), color="tab:orange")
    plt.xlabel("Angle spacing, $\Delta$")
    plt.ylabel("AOA Fisher information")
    # plt.yscale("log")
    # plt.xscale("log")
    plt.show()

    # fig = plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    # ax = fig.add_subplot(111)
    # plt.plot(pos_list, 1/np.sqrt(np.mean(FIthetaN, axis=0)), color="navy")
    # plt.plot(pos_list, 1/np.sqrt(np.mean(FIthetaR, axis=0)), color="tab:orange")
    # plt.xlabel("Angle spacing, $\Delta$")
    # plt.ylabel("AOA CRLB")
    # # plt.yscale("log")
    # # plt.xscale("log")
    # plt.show()

    # plt.plot(coherenceNormalizedN_tot, np.mean(FIthetaN, axis=0), color="tab:purple")
    # plt.xlabel("Normalized coherence")
    # plt.ylabel("Non-RIS AOA Fisher information")
    # plt.show()
    # plt.plot(pos_list, FItheta, color="tab:purple")
    # plt.xlabel("Azimuth spacing")
    # plt.ylabel("Non-RIS AOA Fisher information")
    # plt.show()

    # plt.plot(coherenceNormalizedN_tot, FItheta, color="tab:purple")
    # plt.ylim(0-np.max(FItheta)*0.02, np.max(FItheta)*1.02)
    # plt.xlabel("Normalized coherence")
    # plt.ylabel("Non-RIS AOA Fisher information")
    # plt.show()




