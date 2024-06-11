# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:09:40 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Base functionality added. (13/02/2024)
    v1.1 - Minor bugfixes. Inclusion of rcs as input parameter.
           Introduction of MainSetup method. (23/02/2024)
    v1.2 - Added OMP with hyperparameter choices. (10/06/2024)
"""


import numpy as np
import toml

from scipy.linalg import block_diag
from scipy.ndimage import maximum_filter
from scipy.stats import chi2
from scipy.optimize import minimize

import sys
import os
if os.path.abspath("..")+"/Modules" not in sys.path:
    sys.path.append(os.path.abspath("..")+"/Modules")

from Beamforming import forward_smoothing, smoothing, BartlettSpectrum, \
CaponSpectrum, MUSICSpectrum, PseudoSpectrum_plot, find_peaks
from CompressiveSensing import orthogonal_matching_pursuit
from PP_Sim import PoissonPointProcess, MaternTypeII
from system import channel


class MainEstCore(channel):
    """
    This is a core class for the analysis and numerical experiments of the
    RIS-aided high resolution sensing scenario. The class is inherited 
    from the channel class in the system module.

    The main functionality of this class is to compute loglikelihood,
    compute pseudo-spectra, do channel estimation using the non-RIS signal
    which is used in the prior step, and initializing the scenario.
    """
    def __init__(self, config_file, **kwargs):
        """
        """
        super(MainEstCore, self).__init__(config_file, False, **kwargs)

    # def AngleCali(self, theta_in):
    #     """
    #     Generates calibrated measurement.
    #     """
    #     theta_out = np.copy(theta_in)
    #     while theta_out[0] < -np.pi/2:
    #         theta_out[0] = theta_out[0] + np.pi
    #     while theta_out[0] > np.pi/2:
    #         theta_out[0] = theta_out[0] - np.pi
    #     while theta_out[1] < -np.pi/2:
    #         theta_out[1] = theta_out[1] + np.pi
    #     while theta_out[1] > np.pi/2:
    #         theta_out[1] = theta_out[1] - np.pi
    #     return theta_out

    def loglikelihoodN(self, Covariance, ChParsEst, AlphaEst, sU, W, f, Lsx, Lsy, Lsf):
        """
        Compute the log-likelihood for the estimated channel parameters for
        the non-RIS signal.
        """
        NumberSubarrays = (self.NU[0]-Lsx + 1) * (self.NU[1]-Lsy + 1) * (self.N-Lsf + 1) * self.T2//2
        DimSubarray = Lsx*Lsy*Lsf
        L = ChParsEst.shape[0]
        if L > 0:
            G = np.zeros((DimSubarray, L), dtype=np.complex128)
            for l in range(L):
                G[:, l] = self.gN(ChParsEst[l, 0], ChParsEst[l, 1:3], W, f, Lsx, Lsy, Lsf, 1)
            # PathGains, _ = self.nonRIS_channel_gains(PosEst, pU)
            # P = np.diag(PathGains**2)
            P = np.real(np.outer(AlphaEst, AlphaEst.conj()))
            R = np.dot(G, np.dot(P, G.conj().T)) + self.p_noise/2 * np.eye(DimSubarray)
        else:
            R = self.p_noise/2 * np.eye(DimSubarray)
        R_inv = np.linalg.inv(R)
        R_detsign, R_logdet = np.linalg.slogdet(R)
        LL = - 1/2 * R_logdet - NumberSubarrays/2 * \
            np.real(np.trace(np.dot(R_inv, Covariance)))
        return LL

    def loglikelihoodR(self, Covariance, ChParsEst, AlphaBarEst, sU, theta, phi0, theta0, W, omega, f, Lsf):
        """
        Compute the log-likelihood for the estimated channel parameters for
        the RIS signal.
        """
        NumberSubarrays = self.NU[0] * self.NU[1] * (self.N-Lsf + 1)
        DimSubarray = Lsf*self.T2//2
        L = ChParsEst.shape[0]
        if L > 0:
            G = np.zeros((DimSubarray, L), dtype=np.complex128)
            for l in range(L):
                G[:, l] = self.gR(ChParsEst[l, 0], ChParsEst[l, 1:3], theta[l],
                                  phi0, theta0, W, omega, f, 1, 1, Lsf, self.T2//2)
            # _, PathGains, _ = self.RIS_channel_gains(PosEst, pU)
            # P = np.diag(PathGains**2)
            P = np.real(np.outer(AlphaBarEst, AlphaBarEst.conj()))
            R = np.dot(G, np.dot(P, G.conj().T)) + self.p_noise/2 * np.eye(DimSubarray)
        else:
            R = self.p_noise/2 * np.eye(DimSubarray)
        R_inv = np.linalg.inv(R)
        R_detsign, R_logdet = np.linalg.slogdet(R)
        LL = - 1/2 * R_logdet - NumberSubarrays/2 * \
            np.real(np.trace(np.dot(R_inv, Covariance)))
        return LL

    def loglikelihood(self, CovarianceN, CovarianceR, ChParsEstN, ChParsEstR, AlphaEst, AlphaBarEst,
                      sU, phi0, theta0, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR):
        """
        Compute the log-likelihood for the estimated channel parameters.
        """
        LL = self.loglikelihoodN(CovarianceN, ChParsEstN, AlphaEst, sU, WN, f, LsxN, LsyN, LsfN) \
            + self.loglikelihoodR(CovarianceR, ChParsEstR, AlphaBarEst, sU, ChParsEstN[:, 1:3], phi0, theta0, WR, omega, f, LsfR)
        return LL

    def logprior(self):
        pass

    def gN(self, tau, theta, W, f, Lsx, Lsy, Lsf, Lst):
        """
        Compute the total response vector for the non-RIS signal.

        Inputs:
        -------
            tau : float
            theta : ndarray, size=(2,)

        Output:
        -------
            gN : ndarray, size=(T \cdot N \cdot NY, )
        """
        TimeVec = np.ones(self.T2//2)[:Lst]

        FreqVec = self.DelayVec(tau).flatten()[:Lsf]

        a = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1])
        AMat = np.outer(a, a)
        AngleVec = (np.einsum("ij, i", np.conjugate(W), np.einsum("ij,j->i", AMat, f)).reshape(*self.NU)[:Lsx, :Lsy]).flatten()
        # AngleVec = (a.reshape(*self.NU)[:Lsx, :Lsy]).flatten()

        gN = np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return gN

    def gNTensor(self, tau, az, el, W, f, Lsx, Lsy, Lsf, Lst):
        """
        Compute the total response vector for the non-RIS signal.

        Inputs:
        -------
            tau : ndarray, size=(K_delay)
            az : ndarray, size=(K_az,)
            el : ndarray, size=(K_el,)

        Output:
        -------
            gN : ndarray, size=(K_delay, K_az, K_el, T \cdot N \cdot NY, )
        """
        K_delay, K_az, K_el = len(tau), len(az), len(el)
        TimeVec = np.ones(self.T2//2)[:Lst]  # size=(Lst,)
        FreqVec = self.DelayVec(tau)[:Lsf].T  # size=(K_delay, Lsf)
        # size=(K_az, K_el, NU)
        a = self.ArrayVecTensor_(self.AntPos_UE, az, el)
        AMat = np.einsum("aei,aej->aeij", a, a, optimize="greedy")  # size=(K_az, K_el, NU, NU)
        AngleVec = (np.einsum("ij, aei->aej", np.conjugate(W), np.einsum("aeij,j->aei", AMat, f, optimize="greedy"), optimize="greedy").reshape((K_az, K_el, *self.NU))[:, :, :Lsx, :Lsy]).reshape((K_az, K_el, Lsx*Lsy))
        # AngleVec = (a.reshape((K_az, K_el, self.NU[0], self.NU[1]))[:, :, :Lsx, :Lsy]).reshape((K_az, K_el, Lsx*Lsy))
        gN = np.einsum('t,daefx->daetfx', TimeVec, np.einsum('df,aex->daefx', FreqVec, AngleVec, optimize="greedy"), optimize="greedy").reshape((K_delay, K_az, K_el, -1))
        return gN

    def gR(self, tau_bar, phi, theta, phi0, theta0, W, omega, f, Lsx, Lsy, Lsf, Lst):
        """
        Compute the total response vector for the RIS signal.

        Inputs:
        -------
            tau_bar : float
            phi : ndarray, size=(2,)
            theta : ndarray, size=(2,)
                From prior.
            phi0 : ndarray, size=(2,)
                From prior.
            theta0 : ndarray, size=(2,)
                From prior.

        Output:
        -------
            gR : ndarray, size=(T \cdot N \cdot NY, )
        """
        aR = self.ArrayVec_(self.AntPos_RIS, phi[0], phi[1]) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        TimeVec = np.dot(omega, aR)[:Lst]

        FreqVec = self.DelayVec(tau_bar).flatten()[:Lsf]

        AMat = np.outer(self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]),
                        self.ArrayVec_(self.AntPos_UE, theta[0], theta[1]))
        AngleVec = (np.dot(np.conjugate(W).T, np.dot(AMat, f)).reshape(*self.NU)[:Lsx, :Lsy]).flatten()
        # AngleVec = np.ones((Lsx*Lsy))

        gR = np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return gR

    def gRTensor(self, tau_bar, az, el, theta, phi0, theta0, W, omega, f, Lsx, Lsy, Lsf, Lst):
        """
        Compute the total response vector for the non-RIS signal.

        Inputs:
        -------
            tau_bar : ndarray, size=(K_delay)
            az : ndarray, size=(K_az,)
            el : ndarray, size=(K_el,)

        Output:
        -------
            gR : ndarray, size=(K_delay, K_az, K_el, T \cdot N \cdot NY, )
        """
        K_delay, K_az, K_el = len(tau_bar), len(az), len(el)
        aR = self.ArrayVecTensor_(self.AntPos_RIS, az, el) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])[None, None, :]  # (K_az, K_el, NR)
        # size=(K_az, K_el, Lst,)
        TimeVec = np.einsum("ti, aei -> aet", omega, aR)[:, :, :Lst]
        FreqVec = self.DelayVec(tau_bar)[:Lsf].T  # size=(K_delay, Lsf)
        AMat = np.outer(self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]), self.ArrayVec_(self.AntPos_UE, theta[0], theta[1]))
        AngleVec = (np.dot(np.conjugate(W).T, np.dot(AMat, f)).reshape(*self.NU)[:Lsx, :Lsy]).flatten()
        # AngleVec = np.ones((Lsx*Lsy)) # size=(Lsx*Lsy,)
        gR = np.einsum("aet,dfx->daetfx", TimeVec, np.einsum("df,x->dfx", FreqVec, AngleVec)).reshape((K_delay, K_az, K_el, -1))
        return gR

    def gNR(self, tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR):
        """
        See self.gN and self.gR for details on inputs.

        Output:
        -------
            gNR : ndarray, size=(2 \cdot T \cdot N \cdot NY,)
        """
        gNR = np.hstack((self.gN(tau, theta, WN, f, LsxN, LsyN, LsfN, 1),
                         self.gR(tau_bar, phi, theta, phi0, theta0, WR, omega, f, 1, 1, LsfR, self.T2//2)))
        return gNR

    def dergN(self, tau, theta, W, f, Lsx, Lsy, Lsf, Lst):
        pass

    def dergR(self, tau_bar, phi, theta, phi0, theta0, W, omega, f, Lsx, Lsy, Lsf, Lst):
        pass

    def dergNR(self, tau, theta, tau_bar, phi, phi0, theta0, W, omega, f, LsxN, LsyN, LsfN, LsfR):
        pass

    def PseudoSpectrum_nonRIS(self, Covariance, W, f, Lsx, Lsy, Lsf,
                              delays, az_angles, el_angles,
                              beamformer, M=None, savename="results/spectrum_plots/nonRIS"):
        """
        Compute (and plot) the pseudo-spectrum for the non-RIS signal
        in a specified grid.
        We use dim to denote the sub-array dimension.

        Inputs:
        -------
            Covariance : ndarray, size=(dim, dim)
            delays : ndarray, size=(K_delay,)
                The delays for the grid search.
            az_angles : ndarray, size=(K_az,)
                The azimuth angles for the grid search.
            el_angles : ndarray, size=(K_el,)
                The elevation angles for the grid search.

        Output:
        -------
            P : ndarray, size=(K_delay, K_az, K_el)
                The pseudo-spectrum in the grid.
            ChPars : ndarray, size=(K_delay, K_az, K_el, 3)
                The channel parameters in the grid.
        """
        # Setup search space
        C1, C2, C3 = np.meshgrid(delays, az_angles, el_angles, indexing="ij")
        ChPars = np.stack((C1*1e09, C2, C3), axis=-1)

        # Array response
        gSearch = self.gNTensor(delays, az_angles, el_angles, W, f, Lsx, Lsy, Lsf, 1)

        if beamformer == "CAPON" or beamformer == "MUSIC":
            assert np.linalg.matrix_rank(Covariance) == np.shape(Covariance)[0], "The matrix must be invertible!"

        # Compute pseudo spectrum
        if beamformer == "BARTLETT":
            P = BartlettSpectrum(gSearch, Covariance)
        elif beamformer == "CAPON":
            P = CaponSpectrum(gSearch, Covariance)
        elif beamformer == "MUSIC":
            P = MUSICSpectrum(gSearch, Covariance, M)

        if self.verboseEst is True:
            PseudoSpectrum_plot(delays, az_angles, el_angles,
                                P, savename, title=beamformer)
        return P, ChPars

    def PseudoSpectrum_RIS(self, Covariance, W, omega, f, theta, phi0, theta0, Lsf,
                           delays, az_angles, el_angles,
                           beamformer, M=None, savename="results/spectrum_plots/RIS"):
        """
        Similar to self.PseudoSpectrum_nonRIS.
        """
        # Setup search space
        # K_delay, K_az, K_el = len(delays), len(az_angles), len(el_angles)
        C1, C2, C3 = np.meshgrid(delays, az_angles, el_angles, indexing="ij")
        ChPars = np.stack((C1*1e09, C2, C3), axis=-1)

        # Array response
        # gSearch = np.zeros((K_delay, K_az, K_el, Lsf*self.T2//2), dtype=np.complex128)
        # for idx_delay, tau_bar in enumerate(delays):
        #     for idx_az, az in enumerate(az_angles):
        #         for idx_el, el in enumerate(el_angles):
        #             phi = np.array([az, el])
        #             gSearch[idx_delay, idx_az, idx_el, :] = self.gR(tau_bar, phi, theta, phi0, theta0, W, omega, f, 1, 1, Lsf, self.T2//2)
        gSearch = self.gRTensor(delays, az_angles, el_angles, theta, phi0, theta0, W, omega, f, 1, 1, Lsf, self.T2//2)

        # Spatial smoothing
        if beamformer == "CAPON" or beamformer == "MUSIC":
            assert np.linalg.matrix_rank(Covariance) == np.shape(Covariance)[0], "The matrix must be invertible!"

        # Compute pseudo spectrum
        if beamformer == "BARTLETT":
            P = BartlettSpectrum(gSearch, Covariance)
        elif beamformer == "CAPON":
            P = CaponSpectrum(gSearch, Covariance)
        elif beamformer == "MUSIC":
            P = MUSICSpectrum(gSearch, Covariance, M)

        if self.verboseEst is True:
            PseudoSpectrum_plot(delays, az_angles, el_angles,
                                P, savename, title=beamformer)
        return P, ChPars

    def PseudoSpectrum_joint(self, Y, WN, WR, omega, f, theta, phi0, theta0, LsxN, LsyN, LsfN, LsfR,
                             delaysN, az_anglesN, el_anglesN, delaysR, az_anglesR, el_anglesR,
                             beamformer, M, savename="results/spectrum_plots/temp"):
        """
        Outdated.
        """
        # Setup search space
        K_delayN, K_azN, K_elN = len(delaysN), len(az_anglesN), len(el_anglesN)
        K_delayR, K_azR, K_elR = len(delaysR), len(az_anglesR), len(el_anglesR)
        C1, C2, C3, C4, C5, C6 = np.meshgrid(
            delaysN, az_anglesN, el_anglesN, delaysR, az_anglesR, el_anglesR, indexing="ij")
        ChPars = np.stack((C1*1e09, C2, C3, C4*1e09, C5, C6), axis=-1)

        # Array response
        gSearch = np.zeros((K_delayN, K_azN, K_elN, K_delayR, K_azR,
                           K_elR, LsxN*LsyN*LsfN*LsfR*self.T2//2), dtype=np.complex128)
        for idx_delayN, tau in enumerate(delaysN):
            for idx_azN, azN in enumerate(az_anglesN):
                for idx_elN, elN in enumerate(el_anglesN):
                    for idx_delayR, tau_bar in enumerate(delaysR):
                        for idx_azR, azR in enumerate(az_anglesR):
                            for idx_elR, elR in enumerate(el_anglesR):
                                theta = np.array([azN, elN])
                                phi = np.array([azR, elR])
                                gSearch[idx_delayN, idx_azN, idx_elN, idx_delayR, idx_azR, idx_elR, :] \
                                    = self.gNR(tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR)

        # Estimate covariance matrix
        YN = Y[:self.NU_prod*self.N*self.T2//2]
        RfN = forward_smoothing(YN, LsxN, LsyN, LsfN, 1)
        CovarianceN = smoothing(RfN)
        if beamformer == "CAPON" or beamformer == "MUSIC":
            assert np.linalg.matrix_rank(CovarianceN) == np.shape(
                CovarianceN)[0], "The matrix must be invertible!"

        YR = Y[self.NU_prod*self.N*self.T2//2:]
        RfR = forward_smoothing(YR, 1, 1, LsfR, self.T2//2)
        CovarianceR = RfR  # self.smoothing(RfR)
        if beamformer == "CAPON" or beamformer == "MUSIC":
            assert np.linalg.matrix_rank(CovarianceR) == np.shape(
                CovarianceR)[0], "The matrix must be invertible!"

        Covariance = block_diag(*(CovarianceN, CovarianceR))

        # Compute pseudo spectrum
        if beamformer == "BARTLETT":
            P = BartlettSpectrum(gSearch, Covariance)
        elif beamformer == "CAPON":
            P = CaponSpectrum(gSearch, Covariance)
        elif beamformer == "MUSIC":
            P = MUSICSpectrum(gSearch, Covariance, M)
        return P, ChPars

    def Jacobian_nonRIS(self, eta, W, f, Un, Lsx, Lsy, Lsf):
        pass

    def Jacobian_RIS(self, eta, W, omega, f, Un, Lsf):
        pass

    def Jacobian_joint(self, eta, phi0, theta0, W, omega, f, Un, LsxN, LsyN, LsfN, LsfR):
        pass

    def ObjectiveFunction_nonRIS(self, eta, W, f, Un, Lsx, Lsy, Lsf):
        """
        Compute the MUSIC pseudo-spectrum for the channel parameters eta
        and with noise subspace Un for the non-RIS signal.
        This is used for the optimize choice in the optimization variable, i.e.,
        function optimization rather than grid search.
        """
        tau, theta = eta[0], eta[1:3]
        gN = self.gN(tau*1e-07, theta, W, f, Lsx, Lsy, Lsf, 1)
        prod1 = np.dot(Un.conj().T, gN)
        f_MUSIC = np.real(np.dot(prod1.conj(), prod1))
        return f_MUSIC

    def ObjectiveFunction_RIS(self, eta, W, omega, f, Un, Lsf):
        """
        Same as self.ObjectiveFunction_nonRIS but for RIS signal.
        """
        tau_bar, phi = eta[0], eta[1:3]
        gN = self.gR(tau_bar*1e-07, phi, W, omega, f, 1, 1, Lsf, self.T2//2)
        prod1 = np.dot(Un.conj().T, gN)
        f_MUSIC = np.real(np.dot(prod1.conj(), prod1))
        return f_MUSIC

    def ObjectiveFunction_joint(self, eta, phi0, theta0, WN, WR, omega, f, Un, LsxN, LsyN, LsfN, LsfR):
        """
        Same as self.ObjectiveFunction_nonRIS but for combination of non-RIS
        and RIS signals.
        """
        tau, theta, tau_bar, phi = eta[0], eta[1:3], eta[3], eta[4:]
        gNR = self.gNR(tau*1e-07, theta, tau_bar*1e-07, phi, phi0,
                       theta0, WN, WR, omega, f, LsxN, LsyN, LsfN, LsfR)
        prod1 = np.dot(Un.conj().T, gNR)
        f_MUSIC = np.real(np.dot(prod1.conj(), prod1))
        return f_MUSIC

    def SpectralTest(self, eigs, S, dim, confidence_level, max_=10):
        """
        """
        arithmetic_mean = [S*(dim-d) * np.log(1/(dim-d) * np.sum(eigs[d:])) for d in range(0, max_)]
        geometric_mean = [S * np.sum(np.log(eigs[d:])) for d in range(0, max_)]
        Ld = [(arithmetic_mean[d] - geometric_mean[d]) for d in range(0, max_)]

        dofd = [1/2 * (dim-d) * (dim-d+1) - 1 for d in range(0, max_)]
        p_value = [chi2.cdf(Ld[d], dofd[d]) for d in range(0, max_)]
        M = sum(np.array(p_value) >= confidence_level)
        return M, Ld, p_value

    def EstimateAlpha(self, Y, ChParsEst, W, f, T):
        """
        """
        L_hat = ChParsEst.shape[0]
        if L_hat > 0:
            Gmat = np.zeros((self.NU_prod*self.N*T, L_hat), dtype=np.complex128)
            for l in range(L_hat):
                gVec = self.gN(ChParsEst[l, 0]*1e-09, ChParsEst[l, 1:], W, f, self.NU[0], self.NU[1], self.N, T)
                Gmat[:, l] = gVec
            Gprod = np.dot(Gmat.conj().T, Gmat)
            try:
                GprodInv = np.linalg.inv(Gprod)
                GPseudoInverse = np.dot(GprodInv, Gmat.conj().T)
                AlphaEst = np.dot(GPseudoInverse, Y.flatten())
            except np.linalg.LinAlgError:
                AlphaEst = np.ones(L_hat)*1e-12
        else:
            AlphaEst = np.array([])
        return AlphaEst

    def ChannelEstimationCondM_nonRIS(self, Y, Covariance, sU, prior, W, f, Lsx, Lsy, Lsf,
                                      beamformer, optimization, hpars_grid, M, savename):
        """
        Estimate the channel parameters, the channel coefficient, and the position
        for the non-RIS signal conditioned on having M targets.
        Furthermore, compute the log-likelihood of the estimated channel parameters.
        """
        T = Y.shape[0]
        if optimization == "grid":
            delays = np.linspace(prior["tau_bounds"][0], prior["tau_bounds"][1], hpars_grid["K_delayN"])
            az_angles = np.linspace(prior["theta_bounds"][0], prior["theta_bounds"][1], hpars_grid["K_azN"])
            el_angles = np.linspace(prior["theta_bounds"][2], prior["theta_bounds"][3], hpars_grid["K_elN"])

            P, ChPars = self.PseudoSpectrum_nonRIS(Covariance, W, f, Lsx, Lsy, Lsf, delays,
                                                   az_angles, el_angles, beamformer, M, savename)
            self.priorP, self.priorChPars = P, ChPars
            ChParsEst = np.array([])
            peak_stds = hpars_grid["stdsN"]
            min_peak_stds = 0.5
            while ChParsEst.shape[0] == 0 and peak_stds > min_peak_stds:
                ChParsEst = find_peaks(np.abs(
                    P), ChPars, stds=peak_stds, kernel=hpars_grid["kernelN"], number_of_peaks=M)
                peak_stds -= 0.1
            if ChParsEst.shape[0] == 0:
                tau_prior = (prior["tau_bounds"][0] +
                             prior["tau_bounds"][1])/2*1e09
                az_prior = (prior["theta_bounds"][0] +
                            prior["theta_bounds"][1])/2
                el_prior = (prior["theta_bounds"][2] +
                            prior["theta_bounds"][3])/2
                ChParsEst = np.array([[tau_prior, az_prior, el_prior]])
        elif optimization == "optimize":
            v, U = np.linalg.eigh(Covariance)
            Un = U[:, :-M]

            ### Channel parameter estimation ###
            ChParsEst = np.zeros((0, 3))
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts:
                new = False
                while new is False and attempts < max_attempts:
                    obj_val = 1e10
                    while obj_val > 1e01 and attempts < max_attempts:
                        tau0 = np.random.uniform(
                            prior["tau_bounds"][0], prior["tau_bounds"][1])*1e07
                        theta_az0 = np.random.uniform(
                            prior["theta_bounds"][0], prior["theta_bounds"][1])
                        theta_el0 = np.random.uniform(
                            prior["theta_bounds"][2], prior["theta_bounds"][3])
                        Eta0 = np.array([tau0, theta_az0, theta_el0])
                        res = minimize(self.ObjectiveFunction_nonRIS, Eta0, method="COBYLA", options={"rhobeg": 1e-02},
                                       args=(W, f, Un, Lsx, Lsy, Lsf))
                        obj_val = res["fun"]
                        attempts += 1
                    tau, theta = res["x"][0], res["x"][1:3]
                    # theta = self.AngleCali(theta)
                    Est = np.array([tau, theta[0], theta[1]])
                    diff = np.linalg.norm(Est[None, :]-ChParsEst[:, :], axis=1)
                    if np.all(diff > 0.1):
                        new = True
                        Est = np.array([tau, theta[0], theta[1]])
                        # test if this is the same as was previously found...
                        ChParsEst = np.concatenate(
                            (ChParsEst, np.expand_dims(Est, axis=0)), axis=0)
            ChParsEst[:, 0] = ChParsEst[:, 0]*1e02
        L_hat = ChParsEst.shape[0]
        AlphaEst = self.EstimateAlpha(Y, ChParsEst, W, f, T)

        PosEst = np.zeros((L_hat, 3))
        for idx, z_est in enumerate(ChParsEst):
            p_est = self.ChParsToEuc(*z_est, sU)
            PosEst[idx, :] = p_est

        LL = self.loglikelihoodN(Covariance, ChParsEst, AlphaEst, sU, W, f, Lsx, Lsy, Lsf)
        return ChParsEst, AlphaEst, PosEst, LL

    def ChannelEstimation_nonRIS(self, Y, sU, prior, W, f, Lsx, Lsy, Lsf, order_test, algN, 
                                 beamformer, optimization, confidence_level, residual_threshold, sparsity, hpars_grid, savename):
        """
        For each possible number of targets M, compute the channel parameters,
        the channel coefficients, the position, and the log-likelihood of the
        estimated channel parameters. Run generaliezd log-likelihood ratio tests
        using Wilk's theorem to select the number of targets.
        """
        T, N, NUx, NUy = Y.shape
        Rf = forward_smoothing(Y, Lsx, Lsy, Lsf, 1)
        Covariance = smoothing(Rf)

        if algN == "beamforming":
        # order_test == "eigenvalue_test" or order_test == "MP": # spectral test
            eigs = np.flip(np.linalg.eigh(Covariance)[0])
            S = (NUx-Lsx + 1) * (NUy-Lsy + 1) * (N-Lsf + 1) * (T-1 + 1)
            dim = len(eigs)
            M, _, p_value = self.SpectralTest(eigs, S, dim, confidence_level)
            if M == 0:
                ChParsEst = np.array([])
                AlphaEst = np.array([])
                PosEst = np.array([])
                LL = self.loglikelihoodN(Covariance, np.array([]), np.array([]), sU, None, None, Lsx, Lsy, Lsf)
            else:
                ChParsEst, AlphaEst, PosEst, LL \
                    = self.ChannelEstimationCondM_nonRIS(Y, Covariance, sU, prior, W, f, Lsx, Lsy, Lsf,
                                                         beamformer, optimization, hpars_grid, M, savename)
        elif algN == "OMP":
        # elif order_test == "OMP": # orthogonal matching pursuit
            delays = np.linspace(prior["tau_bounds"][0], prior["tau_bounds"][1], hpars_grid["K_delayN"])
            az_angles = np.linspace(prior["theta_bounds"][0], prior["theta_bounds"][1], hpars_grid["K_azN"])
            el_angles = np.linspace(prior["theta_bounds"][2], prior["theta_bounds"][3], hpars_grid["K_elN"])
            C1, C2, C3 = np.meshgrid(delays, az_angles, el_angles, indexing="ij")
            ChPars = np.stack((C1*1e09, C2, C3), axis=-1)

            try:
                gSearch = np.load("gNSearch.npy")
            except:
                gSearch = np.sqrt(self.p_tx)*self.gNTensor(delays, az_angles, el_angles, W, f, NUx, NUy, N, T)
                np.save("gNSearch.npy", gSearch)
            A = gSearch.reshape((-1, gSearch.shape[-1])).T
            y = Y.flatten()

            x_hat, Lambda = orthogonal_matching_pursuit(A, y,
                                                        stopping_criteria={"eps1": residual_threshold, "eps2": 1e-06, "eps3": 1e-04, "eps4": 1e-02, "sparsity":sparsity},
                                                        plotting={"delays": delays, "az_angles": az_angles, "el_angles": el_angles})
            AlphaEst = x_hat[Lambda]
            ChParsEst = ChPars.reshape((-1, 3))[Lambda]
            L_hat = len(Lambda)
            M = L_hat
            p_value = None

            PosEst = np.zeros((L_hat, 3))
            for idx, z_est in enumerate(ChParsEst):
                p_est = self.ChParsToEuc(*z_est, sU)
                PosEst[idx, :] = p_est
    
            LL = self.loglikelihoodN(Covariance, ChParsEst, AlphaEst, sU, W, f, Lsx, Lsy, Lsf)
        elif order_test == "GLRT": # generalized likelihood ratio test
            ChParsEst_list = list()
            AlphaEst_list = list()
            PosEst_list = list()
            LL_list = list()

            dof = 5

            ChParsEst_list.append(np.array([]))
            AlphaEst_list.append(np.array([]))
            PosEst_list.append(np.array([]))
            LL_list.append(self.loglikelihoodN(Covariance, np.array(
                []), np.array([]), sU, None, None, Lsx, Lsy, Lsf))
            # if self.verboseEst is True:
            #     print("Log-Likelihood: ", LL_list[hyp])

            for M in range(1, order_test):
                ChParsEst, AlphaEst, PosEst, LL \
                    = self.ChannelEstimationCondM_nonRIS(Y, Covariance, sU, prior, W, f, Lsx, Lsy, Lsf,
                                                         beamformer, optimization, hpars_grid, M, savename)
                ChParsEst_list.append(ChParsEst)
                AlphaEst_list.append(AlphaEst)
                PosEst_list.append(PosEst)
                LL_list.append(LL)

            test_statistic_lists = [[-2*(LL_list[hyp-level]-max(LL_list[hyp], LL_list[hyp-level]))
                                         for hyp in range(level, order_test)] for level in range(1, order_test)]
            dof_lists = [[(ChParsEst_list[j+i+1].shape[0]-ChParsEst_list[j].shape[0])*dof for j, test_statistic in enumerate(
                test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]
            p_value_lists = [[1 - chi2.cdf(test_statistic, dof_lists[i][j]) if dof_lists[i][j] > 0 else 1 for j, test_statistic in enumerate(
                test_statistic_list)] for i, test_statistic_list in enumerate(test_statistic_lists)]

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

            # if self.verboseEst is True:
                # print("Channel parameter estimates: \n", ChParsEst)
                # print("Channel coefficient estimates: \n", AlphaEst)
                # print("Position estimates: \n", PosEst)
                # print("Log-Likelihood: ", LL_list[hyp])
                # print(f"Test statistic: {test_statistic:.3f}, \t p-value: {p_value:.2f}")

            ChParsEst = ChParsEst_list[hypEst]
            AlphaEst = AlphaEst_list[hypEst]
            PosEst = PosEst_list[hypEst]
            LL = LL_list[hypEst]

        resN = {"ChParsEstN": ChParsEst, "AlphaEstN": AlphaEst, "PosEstN": PosEst, "LLN": LL, "MN": M, "p_valueN": p_value}

        if self.verboseEst is True:
            # print(f"Chosen hypothesis is M = {hypEst}")
            print("Channel parameter estimates: \n", ChParsEst)
            print("Channel coefficient estimates: \n", AlphaEst)
            print("Position estimates: \n", PosEst)
            # print("Log-Likelihood: ", LL)
        return resN

    def make_prior(self, Phi, sU, res_scale=50):
        """
        Construct the prior as a dictionary containing information on the
        resolution region, the center of the resolution region, and the
        angle to (and from) the RIS.
        """
        pU, _ = sU[:3], sU[3:]

        # phi0, phil = self.RIS_angles(Phi, sU[:3])
        # theta0, thetal = self.nonRIS_angles(Phi, sU)

        # phil_min = np.min(phil, axis=0)
        # phil_max = np.max(phil, axis=0)
        # phi_bounds = (phil_min[0]-0.2, phil_max[0]+0.2, phil_min[1]-0.2, phil_max[1]+0.2)

        # thetal_mean = np.mean(thetal, axis=0)

        # prior = {"thetal": thetal_mean, "phi_bounds": phi_bounds,
        #          "phi0": phi0, "theta0": theta0}

        center = np.mean(Phi, axis=0)

        phi0, phil = self.RIS_angles(np.expand_dims(center, axis=0), pU)
        theta0, thetal = self.nonRIS_angles(np.expand_dims(center, axis=0), sU)
        taul, _ = self.nonRIS_delays(np.expand_dims(center, axis=0), pU)
        _, taul_bar, _ = self.RIS_delays(np.expand_dims(center, axis=0), pU)

        if self.bounds is False:
            ResRegion = self.sensor_resolution_function(center, sU, res_scale)
        else:
            ResRegion = np.zeros((2, 2, 2, 3))
            for idx1, delay in enumerate(self.bounds["tau_bounds"]):
                for idx2, az in enumerate(self.bounds["theta_bounds"][:2]):
                    for idx3, el in enumerate(self.bounds["theta_bounds"][2:]):
                        ResRegion[idx1, idx2, idx3, :] = self.ChParsToEuc(delay*1e09, az, el, sU)
            ResRegion = ResRegion.reshape((-1, 3))

        _, theta_ResRegion = self.nonRIS_angles(ResRegion, sU)
        tau_ResRegion, _ = self.nonRIS_delays(ResRegion, pU)
        _, tau_bar_ResRegion, _ = self.RIS_delays(ResRegion, pU)
        theta_bounds = (np.min(theta_ResRegion, axis=0)[0]-0.1, np.max(theta_ResRegion, axis=0)[0]+0.1,
                        np.min(theta_ResRegion, axis=0)[1]-0.1, np.max(theta_ResRegion, axis=0)[1]+0.1)
        tau_bounds = (np.min(tau_ResRegion)-6e-09, np.max(tau_ResRegion)+6e-09)
        tau_bar_bounds = (np.min(tau_bar_ResRegion)-6e-09, np.max(tau_bar_ResRegion)+6e-09)
        phi_bounds = self.RIS_angle_resolution_region(pU, ResRegion)
        phi_bounds = [phi_bounds[0]-1e-03, phi_bounds[1] + 1e-03, phi_bounds[2]-1e-03, phi_bounds[3]+1e-03]

        A = [np.min(ResRegion, axis=0)[0], np.max(ResRegion, axis=0)[0],
             np.min(ResRegion, axis=0)[1], np.max(ResRegion, axis=0)[1],
             np.min(ResRegion, axis=0)[2], np.max(ResRegion, axis=0)[2]]
        ChA = [np.min(tau_ResRegion), np.max(tau_ResRegion), np.min(theta_ResRegion, axis=0)[0],
               np.max(theta_ResRegion, axis=0)[0], np.min(theta_ResRegion, axis=0)[1], np.max(theta_ResRegion, axis=0)[1]]

        prior = {"taul": taul[0], "thetal": thetal[0], "taul_bar": taul_bar[0], "phil": phil[0],
                 "phi0": phi0, "theta0": theta0,
                 "tau_bounds": tau_bounds, "theta_bounds": theta_bounds,
                 "tau_bar_bounds": tau_bar_bounds, "phi_bounds": phi_bounds,
                 "A": A, "ChA": ChA,
                 "center": np.expand_dims(center, axis=0)}
        return prior

    def InitializeSetup(self, Phi, sU, rcs, hpars):
        """
        Setup the scenario: Construct the prior, if specified simulate the
        target positions, and compute the true channel parameters.
        """
        prior = self.make_prior(Phi, sU, hpars["res_scale"])

        if hpars["simPhi"] != "None":
            if self.verboseEst is True:
                print("Re-initializing setup!")
            if hpars["simPhi"] == "poisson":
                ChPhi = np.array([])
                area = (prior["ChA"][1]-prior["ChA"][0])*(prior["ChA"]
                        [3]-prior["ChA"][2])*(prior["ChA"][5]-prior["ChA"][4])
                # while len(ChPhi) < 1:
                ChPhi = PoissonPointProcess(prior["ChA"], hpars["intensity"]/area).points
            elif hpars["simPhi"] == "binomial":
                ChPhi = PoissonPointProcess(prior["ChA"], None).binomial(hpars["intensity"])
            elif hpars["simPhi"] == "matern":
                ChPhi = np.array([])
                area = (prior["ChA"][1]-prior["ChA"][0])*(prior["ChA"]
                        [3]-prior["ChA"][2])*(prior["ChA"][5]-prior["ChA"][4])
                # while len(ChPhi) < 1:
                ChPhi = MaternTypeII(prior["ChA"]).simulate_MaternTypeII(
                    hpars["intensity"]/area, hpars["interaction_radius"])

            if hpars["simPhi"] == "rcs":
                rcs = np.random.rayleigh(scale=rcs * np.sqrt(2/np.pi))
            else:
                rcs = None
                Phi_rs = ChPhi[:, 0]/2 * self.c
                Phi_azs = ChPhi[:, 1]
                Phi_els = ChPhi[:, 2]
                Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)

        if Phi.shape[0] > 0:
            ChParsDict = self.ChParams(Phi, sU, rcs)
            L = Phi.shape[0]
            ChPars = np.zeros((L, 6))
            for l in range(L):
                ChPars[l] = np.array([ChParsDict["taul"][l]*1e09, ChParsDict["thetal"][l, 0], ChParsDict["thetal"][l, 1],
                                      ChParsDict["taul_bar"][l]*1e09, ChParsDict["phil"][l, 0], ChParsDict["phil"][l, 1]])
        else:
            ChPars = np.array([])
        return Phi, ChPars, rcs, prior

    def PriorSens(self, Phi, sU, rcs, hpars, **kwargs):
        """
        """
        pU, _ = sU[:3], sU[3:]
        try:
            ChPars = kwargs["ChPars"]
            prior = kwargs["prior"]
        except KeyError:
            Phi, ChPars, rcs, prior = self.InitializeSetup(Phi, sU, rcs, hpars)

        if self.bounds is not False:
            # bounds = {"tau_bounds": [90*1e-09, 150*1e-09], "theta_bounds": [0.6, 0.9, 0.7, 1.0]}
            prior["tau_bounds"] = self.bounds["tau_bounds"]
            prior["theta_bounds"] = self.bounds["theta_bounds"]
            # prior["tau_bar_bounds"] = self.bounds["tau_bar_bounds"]
            # prior["phi_bounds"] = self.bounds["phi_bounds"]

        dictget = lambda d, *k: [d[i] for i in k]
        method, order_test, algPrior, beamformer, optimization, LsxN, LsyN, LsfN, cutoff_threshold, confidence_level, residual_thresholdN, sparsity \
            = dictget(hpars, "method", "order_test", "algPrior", "beamformer", "optimization", "LsxN", "LsyN", "LsfN", "cutoff_threshold", "confidence_level", "residual_thresholdN", "sparsity")
        hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                      "stdsN": hpars["stdsN"], "kernelN": hpars["kernelN"]}

        YN, WN, f = self.prior_step_signal_model(Phi, sU, rcs)
        f = f[0, :]
        YN = YN.reshape((self.T1//2, self.N, self.NU[0], self.NU[1]))

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)

        order_test = "eigenvalue_test"
        resPrior = self.ChannelEstimation_nonRIS(YN, sU, prior, WN, f, LsxN, LsyN, LsfN, order_test, algPrior, beamformer, optimization,
                                                 confidence_level, residual_thresholdN, sparsity, hpars_grid, savename="results/spectrum_plots/prior")
        # ChParsEst, AlphaEst, PosEst, LL = resPrior["ChParsEstN"], resPrior["AlphaEstN"], resPrior["PosEstN"], resPrior["LLN"]
        ChParsEst = resPrior["ChParsEstN"]

        M = ChParsEst.shape[0]
        if M == 0:
            tau_bounds = prior["tau_bounds"]
            theta_bounds = prior["theta_bounds"]
            tau_min, tau_max = tau_bounds[0], tau_bounds[1]
            az_min, az_max, el_min, el_max = theta_bounds[0], theta_bounds[1], theta_bounds[2], theta_bounds[3]
        else:
            P, priorChPars = self.priorP, self.priorChPars
            P_abs = np.abs(P)
            max_filter = maximum_filter(P_abs, size=hpars["kernelN"], mode='constant', cval=1e10)
            mask_loc = P_abs >= max_filter
    
            try:
                ref_level = np.flip(np.sort(P_abs[mask_loc]))[M-1]
            except IndexError:  # handling cases where no peak was detected
                ref_level = np.max(P_abs)
            P_abs_norm = P_abs/ref_level
            P_dB_norm = 20*np.log10(P_abs_norm)
            # P_cutoff = np.where(P_dB_norm >= -3, P_dB_norm, -100*np.ones_like(P_dB_norm))
    
            ParsAboveCutoff = priorChPars[P_dB_norm >= cutoff_threshold]
            tau_min, az_min, el_min = np.min(ParsAboveCutoff, axis=0)
            tau_max, az_max, el_max = np.max(ParsAboveCutoff, axis=0)
            tau_min, tau_max = tau_min*1e-09, tau_max*1e-09
            tau_bounds = (tau_min, tau_max)
            theta_bounds = (az_min, az_max, el_min, el_max)
    
            # print(prior)
            # print(tau_min, tau_max, az_min, az_max, el_min, el_max)
            # print(np.min(priorChPars.reshape((-1, 3)), axis=0), np.max(priorChPars.reshape((-1, 3)), axis=0))
    
        ResRegion = np.zeros((2, 2, 2, 3))
        for idx1, delay in enumerate(tau_bounds):
            for idx2, az in enumerate(theta_bounds[:2]):
                for idx3, el in enumerate(theta_bounds[2:]):
                    ResRegion[idx1, idx2, idx3, :] = self.ChParsToEuc(delay*1e09, az, el, sU)
        ResRegion = ResRegion.reshape((-1, 3))

        ParsCenter = [(tau_max + tau_min) / 2, (az_max + az_min) / 2, (el_max + el_min) / 2]
        center = self.ChParsToEuc(ParsCenter[0], ParsCenter[1], ParsCenter[2], sU)

        phi0, phil = self.RIS_angles(np.expand_dims(center, axis=0), pU)
        theta0, thetal = self.nonRIS_angles(np.expand_dims(center, axis=0), sU)
        taul, _ = self.nonRIS_delays(np.expand_dims(center, axis=0), pU)
        _, taul_bar, _ = self.RIS_delays(np.expand_dims(center, axis=0), pU)

        _, tau_bar_ResRegion, _ = self.RIS_delays(ResRegion, pU)
        tau_bar_bounds = (np.min(tau_bar_ResRegion), np.max(tau_bar_ResRegion))
        phi_bounds = self.RIS_angle_resolution_region(pU, ResRegion)
        phi_bounds = [phi_bounds[0], phi_bounds[1], phi_bounds[2], phi_bounds[3]]

        prior = {"taul": taul[0], "thetal": thetal[0], "taul_bar": taul_bar[0], "phil": phil[0],
                 "phi0": phi0, "theta0": theta0,
                 "tau_bounds": tau_bounds, "theta_bounds": theta_bounds,
                 "tau_bar_bounds": tau_bar_bounds, "phi_bounds": phi_bounds,
                 "center": np.expand_dims(center, axis=0)}
        return prior, resPrior

    def MainSetup(self, Phi, sU, rcs, hpars, **kwargs):
        """
        Run high-resolution sensing algorithm.
        """
        try:
            self.bounds = kwargs["bounds"]
        except KeyError:
            pass

        try:
            ChPars = kwargs["ChPars"]
            prior = kwargs["prior"]
        except KeyError:
            Phi, ChPars, rcs, prior = self.InitializeSetup(Phi, sU, rcs, hpars)
        L = Phi.shape[0]

        if self.verboseEst is True:
            print("Targets: \n", Phi)
            triu_idx = np.triu_indices(Phi.shape[0], k=1)
            dists = np.linalg.norm(Phi[None, :, :] - Phi[:, None, :], axis=2)
            print("Distances between targets: \n", dists[triu_idx])
            dist_RIS = np.linalg.norm(self.sR[None, :3] - Phi, axis=1)
            print("Distance to RIS: \n", dist_RIS)
            Fraunhofer_dist = 2*(self.NR[0]*self.lambda_/4)**2/self.lambda_
            print("Fraunhofer distance: \n", Fraunhofer_dist)
            print("Channel parameters: \n", ChPars)

        if hpars["simulate_prior"] is True:
            if self.verboseEst is True:
                print("Constructing prior by doing non-RIS channel estimation!")
            prior, resPrior = self.PriorSens(Phi, sU, rcs, hpars, ChPars=ChPars, prior=prior)
            AlphaPrior = self.alphal
            resPrior = {"AlphaPrior": AlphaPrior, "ChParsEstPrior": resPrior["ChParsEstN"],
                        "AlphaEstPrior": resPrior["AlphaEstN"], "PosEstPrior": resPrior["PosEstN"], "LLPrior": resPrior["LLN"]}
            return Phi, ChPars, rcs, prior, resPrior
        else:
            if self.bounds is not False:
                prior["tau_bounds"] = self.bounds["tau_bounds"]
                prior["theta_bounds"] = self.bounds["theta_bounds"]

                ResRegion = np.zeros((2, 2, 2, 3))
                for idx1, delay in enumerate(prior["tau_bounds"]):
                    for idx2, az in enumerate(prior["theta_bounds"][:2]):
                        for idx3, el in enumerate(prior["theta_bounds"][2:]):
                            ResRegion[idx1, idx2, idx3, :] = self.ChParsToEuc(delay*1e09, az, el, sU)
                ResRegion = ResRegion.reshape((-1, 3))

                _, tau_bar_ResRegion, _ = self.RIS_delays(ResRegion, sU[:3])
                tau_bar_bounds = (np.min(tau_bar_ResRegion), np.max(tau_bar_ResRegion))
                phi_bounds = self.RIS_angle_resolution_region(sU[:3], ResRegion)
                phi_bounds = [phi_bounds[0], phi_bounds[1], phi_bounds[2], phi_bounds[3]]

                prior["tau_bar_bounds"] = tau_bar_bounds
                prior["phi_bounds"] = phi_bounds

                # prior["tau_bar_bounds"] = self.bounds["tau_bar_bounds"]
                # prior["phi_bounds"] = self.bounds["phi_bounds"]
            return Phi, ChPars, rcs, prior, None

    def __call__(self, Phi, sU, rcs, hpars, **kwargs):
        self.bounds = False
        self.verboseEst = True
        return self.PriorSens(Phi, sU, rcs, hpars, **kwargs)
        

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
    mod = MainEstCore(None, **toml_settings)
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
    rcs = np.sqrt(np.array(toml_positions["Phi_rcs"]))

    # =============================================================================
    # Run algorithm
    # =============================================================================
    res = mod(Phi, sU, rcs, toml_estimation)
    # with open("results/temp.pickle", "wb") as file:
    #     pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

    # mod.run_profile()



if __name__ == "__main__":
    main()
