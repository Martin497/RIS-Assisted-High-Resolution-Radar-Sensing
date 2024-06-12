# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:06:11 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implement channel estimation for the main signal model. This includes
           both non-RIS, RIS, parallel, and joint estimation. We have option
           for pseudo-spectrum grid search and pseudo-spectrum numerical
           optimization using scipy.optimize.minimize. Also includes position
           estimation using scipy.optimize.minimize.
           Methods include BARTLETT, CAPON, and MUSIC. (11/12/2023)
    v1.1 - Minor adjustments to functionality. (21/12/2023)
    v1.2 - Addition of hypothesis testing to determine the number of targets
           in the channel estimation. Introduction of intelligent data association
           comparing non-RIS and RIS parameters in the AoA at the RIS domain.
           Added some docstrings. Included some edge case handling. (03/01/2024)
    v1.3 - Important bugfixes in parallel method. (08/01/2024)
    v1.4 - New hypothesis testing method to determine number of targets
           relying on eigenvalues of signal covariance matrix. Extended
           position estimation to weighted least squares, and also
           include position estimation using only RIS signal. Include a new
           option for a prior by simulating the prior sensing step, and
           constructing resolution region based on width of peak in the
           pseudo-spectrum. (02/02/2024)
    v1.5 - Threshold detection with RIS signal. (28/03/2024)
    v1.6 - Implementation of OMP. Threshold detection with OMP. Hyperparameter
           choices related to OMP. (10/06/2024)
"""


import numpy as np
import pickle
import toml

from scipy.optimize import minimize
from scipy.stats import chi2
    
from Beamforming import forward_smoothing, smoothing, find_peaks
from CompressiveSensing import orthogonal_matching_pursuit

from ChAnalysis import ChannelAnalysis
from PositionEstimation_v2 import PosEst
from DataAssociation import data_association_cost, data_association


class ChannelEstimation(ChannelAnalysis):
    """
    Class for doing channel estimation in the two-step protocol.

    Methods
    -------
        __init__ : Initialize settings.

    Attributes
    ----------
        verboseEst : bool
            If true show plots.

    Notation wise we have the following conventions:
    - The single bounce (non-RIS path) delay is tau and the angle is theta with coefficient alpha.
    - The double bounce (RIS path) delay is tau_bar and the angle is phi with coefficient alpha_bar.
    - We use the subscript "az" for the azimuth angle and the subscript "el" for the elevation angle.
    - We use the subscript "hat" for estimates and the subscript "bar" for double bounce parameters.

    A number of notations are shared between methods. Commonly, this includes
    the following list:

    Variable notation
    -----------------
        Phi : ndarray, size=(L, 3)
            The SP positions.
        p : ndarray, size=(3,)
            Position of an SP.
        sU : ndarray, size=(6,)
            The state of the UE.
        pU : ndarray, size=(3,)
            The position of the UE.
        oU : ndarray, size=(3,)
            The orientation of the UE.
        sR : ndarray, size=(6,)
            The state of the RIS.
        pR : ndarray, size=(3,)
            The position of the RIS.
        oR : ndarray, size=(3,)
            The orientation of the RIS.
        eta/eta_hat : ndarray, size=(6,)
            The channel parameters organized as [tau, theta, tau_bar, phi]
            with delays in ns.
        W/WN/WR : ndarray, size=(NU, Ny)
            The combiner matrix.
        f : ndarray, size=(NU, )
            The precoder.
        omega : ndarray, size=(T2//2, NR,)
            The RIS phase profiles.
        Covariance/CovarianceN/CovarianceR : ndarray
            The covariance of the received signal / sub-array of the received signal.
        ChParsEst/ChParsEstN/ChParsEstR : ndarray
            The channel parameter estimates.
        AlphaEst/AlphaBarEst : ndarray
            The estimates of the single bounce / double bounce channel coefficients.
        PosEst/PosEst0/PosEstN : ndarray
            The position estimates.
        Ls... : int
            The dimension of the sub-array along the time, frequency, or spatial axes.
    """
    def __init__(self, config_file=None, verbose=True, bounds=False, **kwargs):
        """
        Initialize class from inheritance.
        """
        super(ChannelEstimation, self).__init__(config_file, **kwargs)
        self.verboseEst = verbose
        self.bounds = bounds
        self.chPosEst = PosEst(None, **kwargs)

    def EstimateAlphaBar(self, Y, ChParsEst, W, omega, f, prior):
        """
        """
        L_hat = ChParsEst.shape[0]
        if L_hat > 0:
            Gmat = np.zeros((self.NU_prod*self.N*self.T2//2, L_hat), dtype=np.complex128)
            for l in range(L_hat):
                gVec = self.gR(ChParsEst[l, 0]*1e-09, ChParsEst[l, 1:],  prior["thetal"], prior["phi0"], prior["theta0"], W, omega, f, self.NU[0], self.NU[1], self.N, self.T2//2)
                Gmat[:, l] = gVec
            Gprod = np.dot(Gmat.conj().T, Gmat)
            try:
                GprodInv = np.linalg.inv(Gprod)
                GPseudoInverse = np.dot(GprodInv, Gmat.conj().T)
                AlphaBarEst = np.dot(GPseudoInverse, Y.flatten())
            except np.linalg.LinAlgError:
                AlphaBarEst = np.ones(L_hat)*1e-12
        else:
            AlphaBarEst = np.array([])
        return AlphaBarEst

    def ChannelEstimationCondM_RIS(self, Y, Covariance, sU, prior, W, omega, f, Lsf,
                                   beamformer, optimization, hpars_grid, M, savename):
        """
        Same as self.ChannelEstimationCondM_nonRIS but for the RIS signal.
        """
        if optimization == "grid":
            if Lsf == 1:
                delays = np.array([(prior["tau_bar_bounds"][0]+prior["tau_bar_bounds"][1])/2])
            else:
                delays = np.linspace(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1], hpars_grid["K_delayR"])

            # az_angles = np.linspace(prior["phi_bounds"][0], prior["phi_bounds"][1], hpars_grid["K_azR"])
            # el_angles = np.linspace(prior["phi_bounds"][2], prior["phi_bounds"][3], hpars_grid["K_elR"])

            delta = 0.1
            az_angles = np.linspace(prior["phi_bounds"][0]+delta*1.3, prior["phi_bounds"][1]-delta*1/2, hpars_grid["K_azR"])
            el_angles = np.linspace(prior["phi_bounds"][2]+delta*1, prior["phi_bounds"][3]-delta*1/2, hpars_grid["K_elR"])

            # az_angles = np.linspace(0, np.pi, hpars_grid["K_azR"])
            # el_angles = np.linspace(0, np.pi/2, hpars_grid["K_elR"])

            P, ChPars = self.PseudoSpectrum_RIS(Covariance, W, omega, f, prior["thetal"], prior["phi0"], prior["theta0"],
                                                Lsf, delays, az_angles, el_angles,
                                                beamformer, M, savename)
            threshold_detection = True
            if threshold_detection is True:
                number_of_peaks = None
            else:
                number_of_peaks = M
            if Lsf == 1:
                ChParsEst = find_peaks(np.abs(P)[0], ChPars[0], stds=hpars_grid["stdsR"], kernel=hpars_grid["kernelR"][1:], number_of_peaks=number_of_peaks)
            else:
                estimate_db_delay = False
                if estimate_db_delay is False:
                    ChParsEst = find_peaks(np.abs(np.sum(P, axis=0)), ChPars[hpars_grid["K_delayR"]//2], stds=hpars_grid["stdsR"], kernel=hpars_grid["kernelR"][1:], number_of_peaks=number_of_peaks)
                else:
                    ChParsEst = find_peaks(np.abs(P), ChPars, stds=hpars_grid["stdsR"], kernel=hpars_grid["kernelR"], number_of_peaks=number_of_peaks)
        elif optimization == "optimize":
            v, U = np.linalg.eigh(Covariance)
            Un = U[:, :-M]

            ### Channel parameter estimation ###
            ChParsEst = np.zeros((0, 3))
            attempts = 0
            while attempts < 20:
                new = False
                while new is False and attempts < 20:
                    obj_val = 1e10
                    while obj_val > 1e01 and attempts < 20:
                        tau_bar0 = np.random.uniform(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1])*1e07
                        phi_az0 = np.random.uniform(prior["phi_bounds"][0], prior["phi_bounds"][1])
                        phi_el0 = np.random.uniform(prior["phi_bounds"][2], prior["phi_bounds"][3])
                        Eta0 = np.array([tau_bar0, phi_az0, phi_el0])
                        res = minimize(self.ObjectiveFunction_RIS, Eta0, method="COBYLA", options={"rhobeg":1e-02},
                                       args=(W, omega, f, Un, Lsf))
                        obj_val = res["fun"]
                        attempts += 1
                    tau_bar, phi = res["x"][0], res["x"][1:3]
                    # phi = self.AngleCali(phi)
                    Est = np.array([tau_bar, phi[0], phi[1]])
                    diff = np.linalg.norm(Est[None, :]-ChParsEst[:, :], axis=1)
                    if np.all(diff > 0.1):
                        new = True
                        Est = np.array([tau_bar, phi[0], phi[1]])
                        ChParsEst = np.concatenate((ChParsEst, np.expand_dims(Est, axis=0)), axis=0) # test if this is the same as was previously found...
            ChParsEst[:, 0] = ChParsEst[:, 0]*1e02
        L_hat = ChParsEst.shape[0]
        AlphaBarEst = self.EstimateAlphaBar(Y, ChParsEst, W, omega, f, prior)

        # Find precision matrix as the equivalent Fisher information
        FIM_USRU = self.FIM_USRU_core(AlphaBarEst, ChParsEst[:, 0], np.repeat(np.expand_dims(prior["thetal"], axis=0), L_hat, axis=0),
                        ChParsEst[:, 1:], prior["theta0"], prior["phi0"], W, omega, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
        EFIM_USRU = self.EFIM(FIM_USRU, ChParsEst.shape[0], type_="RIS")
        reEFIM_USRU = self.rearrange_FIM(EFIM_USRU, type_="RIS")

        PosEst0 = np.zeros((L_hat, 3))
        PosEst = np.zeros((L_hat, 3))
        if L_hat > 0:
            for l in range(L_hat):
                PosEst0[l, :] = prior["center"]
                PosEst[l, :] = self.chPosEst.RISPositionEstimation(sU, self.sR, ChParsEst[l], PosEst0[l], reEFIM_USRU[l*3:(l+1)*3, l*3:(l+1)*3], 12/(prior["tau_bar_bounds"][0]*1e09 - prior["tau_bar_bounds"][1]*1e09)**2)

        LL = self.loglikelihoodR(Covariance, ChParsEst, AlphaBarEst, sU, np.repeat(np.expand_dims(prior["thetal"], axis=0), L_hat, axis=0), prior["phi0"], prior["theta0"], W, omega, f, Lsf)
        return ChParsEst, AlphaBarEst, PosEst, LL

    def ChannelEstimation_RIS(self, Y, sU, prior, W, omega, f, Lsf, order_test, algR,
                              beamformer, optimization, confidence_level, residual_threshold, sparsity, hpars_grid, savename):
        """
        Same as ChannelEstimation_nonRIS but for RIS signal.
        """
        Rf = forward_smoothing(Y, 1, 1, Lsf, self.T2//2)
        Covariance = Rf # smoothing(Rf)
        T, N, NUx, NUy = Y.shape

        if algR == "beamforming":
            eigs = np.flip(np.linalg.eigh(Covariance)[0])
            S = (NUx-1 + 1) * (NUy-1 + 1) * (N-Lsf + 1)
            dim = len(eigs)
            M, _, p_value = self.SpectralTest(eigs, S, dim, confidence_level)
            if M == 0:
                ChParsEst = np.array([])
                AlphaBarEst = np.array([])
                PosEst = np.array([])
                LL = self.loglikelihoodR(Covariance, np.array([]), np.array([]), sU, None, None, None, None, None, None, Lsf)
            else:
                ChParsEst, AlphaBarEst, PosEst, LL \
                    = self.ChannelEstimationCondM_RIS(Y, Covariance, sU, prior, W, omega, f, Lsf,
                                                      beamformer, optimization, hpars_grid, M, savename)
        elif algR == "OMP":
            if Lsf == 1:
                delays = np.array([(prior["tau_bar_bounds"][0]+prior["tau_bar_bounds"][1])/2])
            else:
                delays = np.linspace(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1], hpars_grid["K_delayR"])
            az_angles = np.linspace(prior["phi_bounds"][0], prior["phi_bounds"][1], hpars_grid["K_azR"])
            el_angles = np.linspace(prior["phi_bounds"][2], prior["phi_bounds"][3], hpars_grid["K_elR"])
            C1, C2, C3 = np.meshgrid(delays, az_angles, el_angles, indexing="ij")
            ChPars = np.stack((C1*1e09, C2, C3), axis=-1)

            gSearch = np.sqrt(self.p_tx)*self.gRTensor(delays, az_angles, el_angles, prior["thetal"], prior["phi0"], prior["theta0"], W, omega, f, NUx, NUy, N, T)
            A = gSearch.reshape((-1, gSearch.shape[-1])).T
            y = Y.flatten()

            x_hat, Lambda = orthogonal_matching_pursuit(A, y,
                                                        stopping_criteria={"eps1": residual_threshold, "sparsity":sparsity},
                                                        plotting={"plot": self.verboseEst, "delays": delays, "az_angles": az_angles, "el_angles": el_angles})
            AlphaBarEst = x_hat[Lambda]
            ChParsEst = ChPars.reshape((-1, 3))[Lambda]
            L_hat = len(Lambda)
            M = L_hat
            p_value = None

            # Find precision matrix as the equivalent Fisher information
            FIM_USRU = self.FIM_USRU_core(AlphaBarEst, ChParsEst[:, 0], np.repeat(np.expand_dims(prior["thetal"], axis=0), L_hat, axis=0),
                                          ChParsEst[:, 1:], prior["theta0"], prior["phi0"], W, omega, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
            EFIM_USRU = self.EFIM(FIM_USRU, ChParsEst.shape[0], type_="RIS")
            reEFIM_USRU = self.rearrange_FIM(EFIM_USRU, type_="RIS")

            PosEst0 = np.zeros((L_hat, 3))
            PosEst = np.zeros((L_hat, 3))
            if L_hat > 0:
                for l in range(L_hat):
                    PosEst0[l, :] = prior["center"]
                    PosEst[l, :] = self.chPosEst.RISPositionEstimation(sU, self.sR, ChParsEst[l], PosEst0[l], reEFIM_USRU[l*3:(l+1)*3, l*3:(l+1)*3])

            LL = self.loglikelihoodR(Covariance, ChParsEst, AlphaBarEst, sU, np.repeat(np.expand_dims(prior["thetal"], axis=0), L_hat, axis=0), prior["phi0"], prior["theta0"], W, omega, f, Lsf)
        else:
            ChParsEst_list = list()
            AlphaBarEst_list = list()
            PosEst_list = list()
            LL_list = list()

            dof = 5

            ChParsEst_list.append(np.array([]))
            AlphaBarEst_list.append(np.array([]))
            LL_list.append(self.loglikelihoodR(Covariance, np.array([]), np.array([]), sU,
                                               None, None, None, None, None, None, Lsf))

            for M in range(1, order_test):
                ChParsEst, AlphaBarEst, PosEst, LL \
                    = self.ChannelEstimationCondM_RIS(Y, Covariance, sU, prior, W, omega, f, Lsf,
                                                      beamformer, optimization, hpars_grid, M, savename)

                ChParsEst_list.append(ChParsEst)
                AlphaBarEst_list.append(AlphaBarEst)
                PosEst_list.append(PosEst)
                LL_list.append(LL)
    
            test_statistic_lists = [[-2*(LL_list[hyp-level]-max(LL_list[hyp], LL_list[hyp-level])) for hyp in range(level, order_test)] for level in range(1, order_test)]
            dof_lists = [[(ChParsEst_list[j+i+1].shape[0]-ChParsEst_list[j].shape[0])*dof for j, test_statistic in enumerate(test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]
            p_value_lists = [[1 - chi2.cdf(test_statistic, dof_lists[i][j]) if dof_lists[i][j] > 0 else 1 for j, test_statistic in enumerate(test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]
    
            # accept null hypothesis when p_value > confidence_level for each 
            # subhypothesis in the hypothesis chain.
            hypEst = -1
            level = -10
            for hyp in range(0, order_test-1):
                if level == order_test - hyp:
                    break
                level = 1
                p_value = p_value_lists[level-1][hyp]
                if p_value > confidence_level and hyp <= order_test-2 and hypEst == -1:
                    hypEst = hyp
                while p_value > confidence_level and level < order_test - hyp - 1:
                    level += 1
                    p_value = p_value_lists[level-1][hyp]
                    if level == order_test - hyp - 1:
                        hypEst = hyp
                        break
            if hypEst == -1:
                hypEst = np.argmin(test_statistic_lists[0])

            ChParsEst = ChParsEst_list[hypEst]
            AlphaBarEst = AlphaBarEst_list[hypEst]
            PosEst = PosEst_list[hypEst]
            LL = LL_list[hypEst]

        if self.verboseEst is True:
            # print(f"Chosen hypothesis is M = {hypEst}")
            print("Channel parameter estimates: \n", ChParsEst)
            print("Channel coefficient estimates: \n", AlphaBarEst)
            print("Position estimates: \n", PosEst)
            # print("Log-Likelihood: ", LL)
        resR = {"ChParsEstR": ChParsEst, "AlphaBarEstR": AlphaBarEst, "PosEstR": PosEst, "LLR": LL, "MR": M, "p_valueR": p_value}
        return resR

    def information_fusion(self, YN, YR, sU, WN, WR, omega, f, prior, CovarianceN, CovarianceR, LsxN, LsyN, LsfN, LsfR,
                           ChParsEstN, AlphaEstN, PosEstN, ChParsEstR, AlphaBarEstR, PosEstR):
        """
        """
        square = True

        ### Data Association : Intelligent ###
        L_hatN, L_hatR = ChParsEstN.shape[0], ChParsEstR.shape[0]
        if L_hatN > 0 and L_hatR > 0:
            ChParsEstN_ext = self.ChParamsVec(PosEstN, sU)
            phiN = ChParsEstN_ext[:, 4:6]
            phiR = ChParsEstR[:, 1:3]
            cost_matrix = data_association_cost(phiN, phiR)

            if L_hatN != L_hatR:
                L_hat = max(L_hatN, L_hatR)
                if square is False:
                    ChParsEst = np.zeros((L_hat, 6))
                    AlphaEst = np.zeros(L_hat, dtype=np.complex128)
                    AlphaBarEst = np.zeros(L_hat, dtype=np.complex128)
                    PosEst0 = np.zeros((L_hat, 3))
                    PosEst = np.zeros((L_hat, 3))
                    if L_hatN > L_hatR:
                        association, DAcost = data_association(cost_matrix.T)
                        l = 0
                        for idx1, ass_list in enumerate(association):
                            for idx2, ass in enumerate(ass_list):
                                ChParsEst[l, :3] = ChParsEstN[ass, :]
                                ChParsEst[l, 3:] = ChParsEstR[idx1, :]
                                AlphaEst[l] = AlphaEstN[ass]
                                AlphaBarEst[l] = AlphaBarEstR[idx1]
                                PosEst0[l, :] = PosEstN[ass, :]
                                l += 1
    
                        reEFIM_combined = self.EFIMcombined(AlphaEst, AlphaBarEst, ChParsEst, WN, WR, f, omega, prior)
    
                        for l in range(L_hat):
                            PosEst[l, :] = self.chPosEst.PositionEstimation(sU, self.sR, ChParsEst[l], PosEst0[l])
                    elif L_hatR > L_hatN:
                        association, DAcost = data_association(cost_matrix)
                        l = 0
                        for idx1, ass_list in enumerate(association):
                            for idx2, ass in enumerate(ass_list):
                                ChParsEst[l, :3] = ChParsEstN[idx1, :]
                                ChParsEst[l, 3:] = ChParsEstR[ass, :]
                                AlphaEst[l] = AlphaEstN[idx1]
                                AlphaBarEst[l] = AlphaBarEstR[ass]
                                PosEst0[l, :] = PosEstN[idx1, :]
                                l += 1
    
                        reEFIM_combined = self.EFIMcombined(AlphaEst, AlphaBarEst, ChParsEst, WN, WR, f, omega, prior)
    
                        for l in range(L_hat):
                            PosEst[l, :] = self.chPosEst.PositionEstimation(sU, self.sR, ChParsEst[l], PosEst0[l], reEFIM_combined[l*6:(l+1)*6, l*6:(l+1)*6])
                    # LL = self.loglikelihood(CovarianceN, CovarianceR, ChParsEst[:, :3], ChParsEst[:, 3:], AlphaEst, AlphaBarEst,
                    #                         sU, prior["phi0"], prior["theta0"], WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR)
                elif square is True:
                    if L_hatN > L_hatR:
                        association, DAcost = data_association(cost_matrix.T, square)
                        non_assigned = np.array([a for a in np.arange(L_hatN) if a not in association])

                        ChParsEst = np.zeros((L_hatR, 6))
                        for l in range(L_hatR):
                            ChParsEst[l, :3] = ChParsEstN[association[l], :]
                            ChParsEst[l, 3:] = ChParsEstR[l, :]

                        reEFIM_combined = self.EFIMcombined(AlphaEstN[association], AlphaBarEstR, ChParsEst, WN, WR, f, omega, prior)

                        PosEst = np.zeros((L_hat, 3))
                        for l in range(L_hatR):
                            PosEst[l, :] = self.chPosEst.PositionEstimation(sU, self.sR, ChParsEst[l], PosEstN[association[l]], reEFIM_combined[l*6:(l+1)*6, l*6:(l+1)*6])
                        for i, l in enumerate(range(L_hatR, L_hat)):
                            PosEst[l, :] = PosEstN[non_assigned[i], :]

                        ChParsEst = self.ChParamsVec(PosEst, sU)
                        AlphaEst = self.EstimateAlpha(YN, ChParsEst[:, :3], WN, f, omega.shape[0])
                        AlphaBarEst = self.EstimateAlphaBar(YR, ChParsEst[:, 3:], WR, omega, f, prior)
                    if L_hatR > L_hatN:
                        association, DAcost = data_association(cost_matrix, square)
                        non_assigned = np.array([a for a in np.arange(L_hatR) if a not in association])

                        ChParsEst = np.zeros((L_hatN, 6))
                        for l in range(L_hatN):
                            ChParsEst[l, :3] = ChParsEstN[l, :]
                            ChParsEst[l, 3:] = ChParsEstR[association[l], :]

                        reEFIM_combined = self.EFIMcombined(AlphaEstN, AlphaBarEstR[association], ChParsEst, WN, WR, f, omega, prior)

                        PosEst = np.zeros((L_hat, 3))
                        for l in range(L_hatN):
                            PosEst[l, :] = self.chPosEst.PositionEstimation(sU, self.sR, ChParsEst[l], PosEstN[l], reEFIM_combined[l*6:(l+1)*6, l*6:(l+1)*6])
                        for i, l in enumerate(range(L_hatN, L_hat)):
                            PosEst[l, :] = PosEstR[non_assigned[i], :]

                        ChParsEst = self.ChParamsVec(PosEst, sU)
                        AlphaEst = self.EstimateAlpha(YN, ChParsEst[:, :3], WN, f, omega.shape[0])
                        AlphaBarEst = self.EstimateAlphaBar(YR, ChParsEst[:, 3:], WR, omega, f, prior)
            else:
                L_hat = L_hatN
                association, DAcost = data_association(cost_matrix)

                ChParsEst = np.zeros((L_hat, 6))
                PosEst = np.zeros((L_hat, 3))
                for l in range(L_hat):
                    ChParsEst[l, :3] = ChParsEstN[l, :]
                    ChParsEst[l, 3:] = ChParsEstR[association[l], :]
                AlphaEst = np.copy(AlphaEstN)
                AlphaBarEst = AlphaBarEstR[association]

                reEFIM_combined = self.EFIMcombined(AlphaEst, AlphaBarEst, ChParsEst, WN, WR, f, omega, prior)

                for l in range(L_hat):
                    PosEst[l, :] = self.chPosEst.PositionEstimation(sU, self.sR, ChParsEst[l], PosEstN[l], reEFIM_combined[l*6:(l+1)*6, l*6:(l+1)*6])
            LL = self.loglikelihood(CovarianceN, CovarianceR, ChParsEst[:, :3], ChParsEst[:, 3:], AlphaEst, AlphaBarEst,
                                    sU, prior["phi0"], prior["theta0"], WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR)
        else:
            ChParsEst = np.array([])
            PosEst = np.array([])
            AlphaEst = np.array([])
            AlphaBarEst = np.array([])
            LL = None
        return ChParsEst, AlphaEst, AlphaBarEst, PosEst, LL

    def generalized_likelihood_ratio_test(self, YN, YR, sU, WN, WR, omega, f, prior, CovarianceN, CovarianceR,
                                          LsxN, LsyN, LsfN, LsfR, hyp_max, confidence_level,
                                          beamformer, optimization, hpars_grid, savenameN, savenameR):
        """
        Do channel estimation using the generalized likelihood ratio test to
        choose the number of targets.
        """
        dof = 5

        ChParsEstN_list = list()
        AlphaEstN_list = list()
        PosEstN_list = list()
        LLN_list = list()

        ChParsEstR_list = list()
        AlphaBarEstR_list = list()
        PosEstR_list = list()
        LLR_list = list()

        ChParsEstN_list.append(np.array([]))
        AlphaEstN_list.append(np.array([]))
        PosEstN_list.append(np.array([]))
        LLN_list.append(self.loglikelihoodN(CovarianceN, np.array([]), np.array([]), sU, None, None, LsxN, LsyN, LsfN))

        ChParsEstR_list.append(np.array([]))
        AlphaBarEstR_list.append(np.array([]))
        PosEstR_list.append(np.array([]))
        LLR_list.append(self.loglikelihoodR(CovarianceR, np.array([]), np.array([]), sU,
                                            None, None, None, None, None, None, LsfR))

        for M in range(1, hyp_max):
            ChParsEstN, AlphaEstN, PosEstN, LLN \
                = self.ChannelEstimationCondM_nonRIS(YN, CovarianceN, sU, prior, WN, f, LsxN, LsyN, LsfN,
                                                     beamformer, optimization, hpars_grid, M, savenameN)
            ChParsEstN_list.append(ChParsEstN)
            AlphaEstN_list.append(AlphaEstN)
            PosEstN_list.append(PosEstN)
            LLN_list.append(LLN)

            ChParsEstR, AlphaBarEstR, PosEstR, LLR \
                = self.ChannelEstimationCondM_RIS(YR, CovarianceR, sU, prior, WR, omega, f, LsfR,
                                                  beamformer, optimization, hpars_grid, M, savenameR)

            ChParsEstR_list.append(ChParsEstR)
            AlphaBarEstR_list.append(AlphaBarEstR)
            PosEstR_list.append(PosEstR)
            LLR_list.append(LLR)

        ### Estimate number of targets : Hypothesis testing ###
        LL_list = [lr + ln for lr, ln in zip(LLR_list, LLN_list)]

        card_list = [(estN.shape[0], estR.shape[0]) for estN, estR in zip(ChParsEstN_list, ChParsEstR_list)]
        test_statistic_lists = [[-2*(LL_list[hyp-level]-max(LL_list[hyp], LL_list[hyp-level])) for hyp in range(level, hyp_max)] for level in range(1, hyp_max)]
        dof_lists = [[(max(card_list[j+i+1])-max(card_list[j]))*dof for j, test_statistic in enumerate(test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]
        p_value_lists = [[1 - chi2.cdf(test_statistic, dof_lists[i][j]) if dof_lists[i][j] > 0 else 1 for j, test_statistic in enumerate(test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]

        # accept null hypothesis when p_value > confidence_level for each 
        # subhypothesis in the hypothesis chain.
        hypEst = -1
        level = -10
        for hyp in range(0, hyp_max-1):
            if level == hyp_max - hyp:
                break
            level = 1
            p_value = p_value_lists[level-1][hyp]
            if p_value > confidence_level and hyp <= hyp_max-2 and hypEst == -1:
                hypEst = hyp
            while p_value > confidence_level and level < hyp_max - hyp - 1:
                level += 1
                p_value = p_value_lists[level-1][hyp]
                if level == hyp_max - hyp - 1:
                    hypEst = hyp
                    break
        if hypEst == -1:
            hypEst = 0

        ChParsEstN = ChParsEstN_list[hypEst]
        AlphaEstN = AlphaEstN_list[hypEst]
        PosEstN = PosEstN_list[hypEst]
        LLN = LLN_list[hypEst]
        if self.verboseEst is True:
            print("Non-RIS channel parameter estimates: \n", ChParsEstN)
            print("Non-RIS channel coefficient estimates: \n", AlphaEstN)
            print("Non-RIS position estimates: \n", PosEstN)

        ChParsEstR = ChParsEstR_list[hypEst]
        AlphaBarEstR = AlphaBarEstR_list[hypEst]
        PosEstR = PosEstR_list[hypEst]
        LLR = LLR_list[hypEst]
        if self.verboseEst is True:
            print("RIS channel parameter estimates: \n", ChParsEstR)
            print("RIS channel coefficient estimates: \n", AlphaBarEstR)

        # LL = LL_list[hypEst]
        return ChParsEstN, AlphaEstN, PosEstN, LLN, ChParsEstR, AlphaBarEstR, PosEstR, LLR

    def ChannelEstimation_joint(self, YN, YR, sU, prior, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR,
                                order_test, algN, algR, beamformer, optimization, confidence_level, residual_thresholdN, residual_thresholdR, sparsity,
                                hpars_grid, savenameN, savenameR):
        """
        Run channel estimation for the non-RIS signal and the RIS signal
        and estimate the number of targets.
        Follow this up with data association by comparing the phi angles.
        Combine the estimates and compute the final estimate of the position(s).
        """
        RfN = forward_smoothing(YN, LsxN, LsyN, LsfN, 1)
        CovarianceN = smoothing(RfN)

        CovarianceR = forward_smoothing(YR, 1, 1, LsfR, self.T2//2)

        if order_test == "eigenvalue_test" or order_test == "OMP": # MAIN
            resN = self.ChannelEstimation_nonRIS(YN, sU, prior, WN, f, LsxN, LsyN, LsfN, order_test, algN, beamformer, optimization, confidence_level, residual_thresholdN, sparsity, hpars_grid, savenameN)
            resR  = self.ChannelEstimation_RIS(YR, sU, prior, WR, omega, f, LsfR, order_test, algR, beamformer, optimization, confidence_level, residual_thresholdR, sparsity, hpars_grid, savenameR)
            ChParsEstN, AlphaEstN, PosEstN = resN["ChParsEstN"], resN["AlphaEstN"], resN["PosEstN"]
            ChParsEstR, AlphaBarEstR, PosEstR = resR["ChParsEstR"], resR["AlphaBarEstR"], resR["PosEstR"]
        elif order_test == "joint_eigenvalue_test": # Outdated
            eigsN = np.flip(np.linalg.eigh(CovarianceN)[0])
            eigsR = np.flip(np.linalg.eigh(CovarianceR)[0])

            T, N, NUx, NUy = YN.shape
            SN = (NUx-LsxN + 1) * (NUy-LsyN + 1) * (N-LsfN + 1) * (T-1 + 1)
            SR = (NUx-1 + 1) * (NUy-1 + 1) * (N-LsfR + 1)

            dimN = len(eigsN)
            dimR = len(eigsR)

            M, p_value = self.SpectralTestJoint(eigsN, eigsR, SN, SR, dimN, dimR, confidence_level)
            if M == 0:
                ChParsEstN = np.array([])
                AlphaEstN = np.array([])
                PosEstN = np.array([])
                LLN = self.loglikelihoodN(CovarianceN, np.array([]), np.array([]), sU, None, None, LsxN, LsyN, LsfN)
                ChParsEstR = np.array([])
                AlphaBarEstR = np.array([])
                PosEstR = np.array([])
                LLR = self.loglikelihoodR(CovarianceR, np.array([]), np.array([]), sU, None, None, None, None, None, None, LsfR)
            else:
                ChParsEstN, AlphaEstN, PosEstN, LLN \
                    = self.ChannelEstimationCondM_nonRIS(YN, CovarianceN, sU, prior, WN, f, LsxN, LsyN, LsfN,
                                                         beamformer, optimization, hpars_grid, M, savenameN)
                ChParsEstR, AlphaBarEstR, PosEstR, LLR \
                    = self.ChannelEstimationCondM_RIS(YR, CovarianceR, sU, prior, WR, omega, f, LsfR,
                                                      beamformer, optimization, hpars_grid, M, savenameR)
            resN = {"ChParsEstN": ChParsEstN, "AlphaEstN": AlphaEstN, "PosEstN": PosEstN, "LLN": LLN}
            resR = {"ChParsEstR": ChParsEstR, "AlphaBarEstR": AlphaBarEstR, "PosEstR": PosEstR, "LLR": LLR}
        elif order_test == "generalized_likelihood_ratio_test": # Outdated
            hyp_max = 4
            ChParsEstN, AlphaEstN, PosEstN, LLN, ChParsEstR, AlphaBarEstR, PosEstR, LLR \
                = self.generalized_likelihood_ratio_test(YN, YR, sU, WN, WR, omega, f, prior, CovarianceN, CovarianceR,
                    LsxN, LsyN, LsfN, LsfR, hyp_max, confidence_level, beamformer, optimization, hpars_grid, savenameN, savenameR)
            resN = {"ChParsEstN": ChParsEstN, "AlphaEstN": AlphaEstN, "PosEstN": PosEstN, "LLN": LLN}
            resR = {"ChParsEstR": ChParsEstR, "AlphaBarEstR": AlphaBarEstR, "PosEstR": PosEstR, "LLR": LLR}

        ChParsEst, AlphaEst, AlphaBarEst, PosEst, LL = self.information_fusion(YN, YR, sU, WN, WR, omega, f, prior,
            CovarianceN, CovarianceR, LsxN, LsyN, LsfN, LsfR, ChParsEstN, AlphaEstN, PosEstN, ChParsEstR, AlphaBarEstR, PosEstR)
        resFused = {"ChParsEst": ChParsEst, "AlphaEst": AlphaEst,
                    "AlphaBarEst": AlphaBarEst, "PosEst": PosEst, "LL": LL}

        if self.verboseEst is True:
            # print(f"Chosen hypothesis is M = {ChParsEst.shape[0]}")
            print("Non-RIS channel parameter estimates: \n", ChParsEstN)
            print("Non-RIS position estimates: \n", PosEstN)
            print("RIS channel parameter estimates: \n", ChParsEstR)
            print("RIS position estimates: \n", PosEstR)
            print("Joint channel parameter estimates: \n", ChParsEst)
            print("Joint position estimates: \n", PosEst)
        return resN, resR, resFused

    def ThresholdDetectionRIS(self, Y, Covariance, sU, prior, W, omega, f, Lsf, hpars_grid, M):
        """
        Same as self.ChannelEstimationCondM_nonRIS but for the RIS signal.
        """
        if Lsf == 1:
            delays = np.array([(prior["tau_bar_bounds"][0]+prior["tau_bar_bounds"][1])/2])
        else:
            delays = np.linspace(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1], hpars_grid["K_delayR"])

        delta = 0.1
        az_angles = np.linspace(prior["phi_bounds"][0]+delta*1.3, prior["phi_bounds"][1]-delta*1/2, hpars_grid["K_azR"])
        el_angles = np.linspace(prior["phi_bounds"][2]+delta*1, prior["phi_bounds"][3]-delta*1/2, hpars_grid["K_elR"])

        P, ChPars = self.PseudoSpectrum_RIS(Covariance, W, omega, f, prior["thetal"], prior["phi0"], prior["theta0"],
                                            Lsf, delays, az_angles, el_angles, "MUSIC", min(max(2, M), 4))
        if Lsf == 1:
            local_max_val, local_max_pars, muP, stdP = find_peaks(np.abs(P)[0], ChPars[0], stds=hpars_grid["stdsR"], kernel=hpars_grid["kernelR"][1:], return_local_maxima=True)
        else:
            local_max_val, local_max_pars, muP, stdP = find_peaks(np.abs(np.sum(P, axis=0)), ChPars[hpars_grid["K_delayR"]//2], stds=hpars_grid["stdsR"], kernel=hpars_grid["kernelR"][1:], return_local_maxima=True)
        return local_max_val, local_max_pars, muP, stdP

    def SpectralTestJoint(self, eigsN, eigsR, SN, SR, dimN, dimR, confidence_level, max_=10):
        """
        """
        _, _, p_value = self.SpectralStatisticJoint(eigsN, eigsR, SN, SR, dimN, dimR, max_)
        M = sum(np.array(p_value) >= confidence_level)
        return M, p_value

    def SpectralStatisticJoint(self, eigsN, eigsR, SN, SR, dimN, dimR, max_=10):
        """
        """
        ### Non-RIS ###
        arithmetic_meanN = [SN*(dimN-d) * np.log(1/(dimN-d) * np.sum(eigsN[d:])) for d in range(0, max_)]
        geometric_meanN = [SN * np.sum(np.log(eigsN[d:])) for d in range(0, max_)]
        LdN = [(arithmetic_meanN[d] - geometric_meanN[d]) for d in range(0, max_)]
        dofdN = [1/2 * (dimN-d) * (dimN-d+1) - 1 for d in range(0, max_)]
        p_valueN = [chi2.cdf(LdN[d], dofdN[d]) for d in range(0, max_)]

        ### RIS ###
        arithmetic_meanR = [SR*(dimR-d) * np.log(1/(dimR-d) * np.sum(eigsR[d:])) for d in range(0, max_)]
        geometric_meanR = [SR * np.sum(np.log(eigsR[d:])) for d in range(0, max_)]
        LdR = [(arithmetic_meanR[d] - geometric_meanR[d]) for d in range(0, max_)]
        dofdR = [1/2 * (dimR-d) * (dimR-d+1) - 1 for d in range(0, max_)]
        p_valueR = [chi2.cdf(LdR[d], dofdR[d]) for d in range(0, max_)]

        ### Joint ###
        Ld = [LdN[d] + LdR[d] for d in range(0, max_)]
        dofd = [a + b for a, b in zip(dofdN, dofdR)]
        p_value = [chi2.cdf(Ld[d], dofd[d]) for d in range(0, max_)]
        return p_valueN, p_valueR, p_value

    def SpectralDetection(self, Phi, sU, rcs, hpars, **kwargs):
        """
        Run detection protocol ONLY.
        """
        dictget = lambda d, *k: [d[i] for i in k]
        LsxN, LsyN, LsfN, LsfR = dictget(hpars, "LsxN", "LsyN", "LsfN", "LsfR")

        # Make prior
        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        YN, YR, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]
        YN, YR = YN.reshape((self.T2//2, self.N, self.NU[0], self.NU[1])), YR.reshape((self.T2//2, self.N, self.NU[0], self.NU[1]))

        # Make detection
        RfN = forward_smoothing(YN, LsxN, LsyN, LsfN, 1)
        CovarianceN = smoothing(RfN)

        CovarianceR = forward_smoothing(YR, 1, 1, LsfR, self.T2//2)

        eigsN = np.flip(np.linalg.eigh(CovarianceN)[0])
        eigsR = np.flip(np.linalg.eigh(CovarianceR)[0])

        T, N, NUx, NUy = YN.shape
        SN = (NUx-LsxN + 1) * (NUy-LsyN + 1) * (N-LsfN + 1) * (T-1 + 1)
        SR = (NUx-1 + 1) * (NUy-1 + 1) * (N-LsfR + 1)

        dimN = len(eigsN)
        dimR = len(eigsR)

        p_valueN, p_valueR, p_value = self.SpectralStatisticJoint(eigsN, eigsR, SN, SR, dimN, dimR)
        # confidence_level = np.logspace(-10, 0, 1000)
        # confidence_level = np.linspace(0, 1, 1001)

        # MarrN = np.sum(np.array(p_valueN)[:, None] > confidence_level[None, :], axis=0).astype(np.int8)
        # MarrR = np.sum(np.array(p_valueR)[:, None] > confidence_level[None, :], axis=0).astype(np.int8)
        # Marr = np.sum(np.array(p_value)[:, None] > confidence_level[None, :], axis=0).astype(np.int8)

        resSpectral = {"p_value": p_value, "p_valueN": p_valueN, "p_valueR": p_valueR}
        return resSpectral

    def ThresholdDetection(self, Phi, sU, rcs, hpars, **kwargs):
        """
        Run detection protocol ONLY.
        """
        order_test = hpars["order_test"]

        # Make prior
        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        YN, YR, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]
        YN, YR = YN.reshape((self.T2//2, self.N, self.NU[0], self.NU[1])), YR.reshape((self.T2//2, self.N, self.NU[0], self.NU[1]))

        dictget = lambda d, *k: [d[i] for i in k]
        LsxN, LsyN, LsfN, LsfR = dictget(hpars, "LsxN", "LsyN", "LsfN", "LsfR")

        T, N, NUx, NUy = YN.shape

        if order_test == "eigenvalue_test":
            # Make detection
            RfN = forward_smoothing(YN, LsxN, LsyN, LsfN, 1)
            CovarianceN = smoothing(RfN)

            CovarianceR = forward_smoothing(YR, 1, 1, LsfR, self.T2//2)

            eigsN = np.flip(np.linalg.eigh(CovarianceN)[0])
            eigsR = np.flip(np.linalg.eigh(CovarianceR)[0])

            T, N, NUx, NUy = YN.shape
            SN = (NUx-LsxN + 1) * (NUy-LsyN + 1) * (N-LsfN + 1) * (T-1 + 1)
            SR = (NUx-1 + 1) * (NUy-1 + 1) * (N-LsfR + 1)

            dimN = len(eigsN)
            dimR = len(eigsR)

            p_valueN, p_valueR, p_value = self.SpectralStatisticJoint(eigsN, eigsR, SN, SR, dimN, dimR)

            MR = sum(np.array(p_valueR) > hpars["confidence_level"])
            hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                          "K_delayR": hpars["K_delayR"], "K_azR": hpars["K_azR"], "K_elR": hpars["K_elR"],
                          "stdsN": hpars["stdsN"], "kernelN": hpars["kernelN"],
                          "stdsR": hpars["stdsR"], "kernelR": hpars["kernelR"]}
            local_max_val, local_max_pars, muP, stdP = self.ThresholdDetectionRIS(YR, CovarianceR, sU, prior, WR, omega, f, LsfR, hpars_grid, MR)

            resSpectral = {"p_value": p_value, "p_valueN": p_valueN, "p_valueR": p_valueR,
                           "local_max_val": local_max_val, "local_max_pars": local_max_pars, "muP": muP, "stdP": stdP}
            return resSpectral
        elif order_test == "OMP":
            hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                          "K_delayR": hpars["K_delayR"], "K_azR": hpars["K_azR"], "K_elR": hpars["K_elR"]}
            # =============================================================================
            # RIS
            # =============================================================================
            delays = np.linspace(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1], hpars_grid["K_delayR"])
            az_angles = np.linspace(prior["phi_bounds"][0], prior["phi_bounds"][1], hpars_grid["K_azR"])
            el_angles = np.linspace(prior["phi_bounds"][2], prior["phi_bounds"][3], hpars_grid["K_elR"])

            gRSearch = np.sqrt(self.p_tx)*self.gRTensor(delays, az_angles, el_angles, prior["thetal"], prior["phi0"], prior["theta0"], WR, omega, f, NUx, NUy, N, T)
            AR = gRSearch.reshape((-1, gRSearch.shape[-1])).T
            yR = YR.flatten()

            errorR = self.DetectionOMP(AR, yR, sparsity=hpars["sparsity"]-1)

            # =============================================================================
            # Non-RIS
            # =============================================================================
            delays = np.linspace(prior["tau_bounds"][0], prior["tau_bounds"][1], hpars_grid["K_delayN"])
            az_angles = np.linspace(prior["theta_bounds"][0], prior["theta_bounds"][1], hpars_grid["K_azN"])
            el_angles = np.linspace(prior["theta_bounds"][2], prior["theta_bounds"][3], hpars_grid["K_elN"])

            gNSearch = np.sqrt(self.p_tx)*self.gNTensor(delays, az_angles, el_angles, WN, f, NUx, NUy, N, T)
            AN = gNSearch.reshape((-1, gNSearch.shape[-1])).T
            yN = YN.flatten()

            errorN = self.DetectionOMP(AN, yN, sparsity=hpars["sparsity"]-1)

            resOMP = {"errorN": errorN, "errorR": errorR}
            return resOMP

    def DetectionOMP(self, A, y, sparsity):
        """
        """
        m, n = np.shape(A)
        error = []
        error.append(np.linalg.norm(y))
        Lambda = []
        r = y
        t = 0
        while True:
            receive_power = np.abs(np.einsum("mn,m->n", A.conj(), r, optimize="greedy"))
            J = np.argmax(receive_power)
            if J in Lambda:
                break
            Lambda.append(J)
            x_hat_update = np.zeros(n, dtype=np.complex128)
            x_hat_update[Lambda] = np.dot(np.linalg.pinv(A[:, Lambda]), y)
            r = y - np.dot(A[:, Lambda], x_hat_update[Lambda])
            t += 1
            error.append(np.linalg.norm(r))
            if t == sparsity:
                break
        return error

    def ProcessSignal(self, YN, YR, WN, WR, omega, f, sU, prior, hpars):
        """
        Run the estimation algorithm using the observed signals and taking
        the prior into account.
        """
        dictget = lambda d, *k: [d[i] for i in k]
    
        method, order_test, algN, algR, beamformer, optimization, LsxN, LsyN, LsfN, LsfR, confidence_level, residual_thresholdN, residual_thresholdR, sparsity \
            = dictget(hpars, "method", "order_test", "algN", "algR", "beamformer", "optimization", "LsxN", "LsyN", "LsfN", "LsfR", "confidence_level", "residual_thresholdN", "residual_thresholdR", "sparsity")
        hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                      "K_delayR": hpars["K_delayR"], "K_azR": hpars["K_azR"], "K_elR": hpars["K_elR"],
                      "stdsN": hpars["stdsN"], "kernelN": hpars["kernelN"],
                      "stdsR": hpars["stdsR"], "kernelR": hpars["kernelR"]}

        if optimization == "optimize":
            assert beamformer == "MUSIC", "Restricted support..."
        # if method == "joint":
        #     assert optimization == "optimize" and beamformer == "MUSIC", "Restricted support..."
        # assert method == "nonRIS" or method == "joint", "Restricted support..."

        if method == "nonRIS": # (tau, theta)
            CovarianceN, ChParsEst, AlphaEst, PosEst, LL \
                = self.ChannelEstimation_nonRIS(YN, sU, prior, WN, f, LsxN, LsyN, LsfN,
                                                order_test, algN, beamformer, optimization, confidence_level, residual_thresholdN, sparsity,
                                                hpars_grid, savename="results/spectrum_plots/nonRIS")
            res = {"ChParsEst": ChParsEst, "AlphaEst": AlphaEst, "PosEst": PosEst, "LL": LL}
        elif method == "RIS": # (bar{tau}, phi)
            CovarianceR, ChParsEst, AlphaBarEst, PosEst, LL \
                = self.ChannelEstimation_RIS(YR, sU, prior, WR, omega, f, LsfR, order_test, algR,
                                             beamformer, optimization, confidence_level, residual_thresholdR, sparsity,
                                             hpars_grid, savename="results/spectrum_plots/RIS")
            res = {"ChParsEst": ChParsEst, "AlphaBarEst": AlphaBarEst, "PosEst": PosEst, "LL": LL}
        elif method == "joint": # (tau, theta, bar{tau}, phi)
            resN, resR, resFused \
                = self.ChannelEstimation_joint(YN, YR, sU, prior, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR,
                                               order_test, algN, algR, beamformer, optimization, confidence_level, residual_thresholdN, residual_thresholdR, sparsity,
                                               hpars_grid, savenameN="results/spectrum_plots/nonRIS", savenameR="results/spectrum_plots/RIS")
            res = {**resN, **resR, **resFused}
        return res

    def HiResSens(self, Phi, sU, rcs, hpars, run_fisher=False, run_detection=False, **kwargs):
        """
        Run high-resolution sensing algorithm.
        """
        Phi, ChPars, rcs, prior, resPrior = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)
        L = Phi.shape[0]

        # Simulate signal model
        YN, YR, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]
        YN, YR = YN.reshape((self.T2//2, self.N, self.NU[0], self.NU[1])), YR.reshape((self.T2//2, self.N, self.NU[0], self.NU[1]))

        if L > 0:
            if run_fisher is True:
                resFisher = self.mainFisher(Phi, sU, self.alphal, ChPars[:, 0], ChPars[:, 1:3], self.alphal_bar,
                                            ChPars[:, 3], ChPars[:, 4:], prior["theta0"], prior["phi0"], WN, WR,
                                            omega, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
            else:
                resFisher = dict()
            if run_detection is True:
                resDetect = self.detection_probability_greedy(YN, YR, Phi, sU, rcs, WN, WR, omega, f)
            else:
                resDetect = dict()
        else:
            resFisher = dict()
            resDetect = dict()

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        if self.verboseEst is True:
            print("Beginning high-resolution sensing!")
        res = self.ProcessSignal(YN, YR, WN, WR, omega, f, sU, prior, hpars)
        res = {**res, **resFisher, **resDetect}
        if Phi.shape[0] != 0:
            res.update({"Phi": Phi, "ChPars": ChPars, "Alpha": self.alphal, "AlphaBar": self.alphal_bar})
        else:
            res.update({"Phi": Phi, "ChPars": ChPars, "Alpha": np.array([]), "AlphaBar": np.array([])})            
        res.update({"prior": prior})
        if hpars["simulate_prior"] is True:
            res.update(resPrior)
        return res

    def __call__(self, Phi, sU, rcs, hpars, **kwargs):
        return self.HiResSens(Phi, sU, rcs, hpars, **kwargs)


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
    mod = ChannelEstimation(None, True, False, **toml_settings)
    sU = np.array(toml_settings["sU"])

    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    # =============================================================================
    # Read positions
    # =============================================================================
    Phi_taus = np.array(toml_positions["Phi_taus"])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array(toml_positions["Phi_azs"])
    Phi_els = np.array(toml_positions["Phi_els"])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    print(Phi)
    rcs = np.sqrt(np.array(toml_positions["Phi_rcs"]))

    # =============================================================================
    # Run algorithm
    # =============================================================================
    res = mod.HiResSens(Phi, sU, rcs, toml_estimation, run_fisher=False, run_detection=False, bounds=bounds)
    with open("results/temp.pickle", "wb") as file:
        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

