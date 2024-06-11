# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:19:42 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Base functionality to compute the Fisher information matrix for the
           USU and USRU signals. Includes also EFIM, CRLB, EFIM rearranging
           and transformation to Euclidean coordinates. (21/12/2023)
    v1.1 - Inclusion of additional methods, allowing for more flexible use. (02/02/2024)
    v1.2 - Moving to ChAnalysis.py module. (23/02/2024)
    v1.3 - Minor updates, clean-up, debug lower bound. (10/06/2024)
"""


import numpy as np
import matplotlib.pyplot as plt
import toml

from scipy.linalg import block_diag, sqrtm
from scipy.stats import ncx2
from sklearn.metrics import auc

from MainEstimationCore import MainEstCore


class ChannelAnalysis(MainEstCore):
    """
    This class is used for the analysis and numerical experiments of the
    RIS-aided high resolution sensing scenario. The class is inherited 
    from the MainEstCore class in the MainEstimationCore module.

    The main functionality of this class is to do Fisher analysis and
    compute theoretical detection probabilities.
    """
    def __init__(self, config_file, **kwargs):
        """
        """
        super(ChannelAnalysis, self).__init__(config_file, **kwargs)

    def derArrayMat(self, arg, AntPos, az, el):
        """
        """
        ArrayVec = self.ArrayVec_(AntPos, az, el)
        pdvArrayVec = self.derArrayVec(arg, AntPos, az, el)
        term = np.outer(pdvArrayVec, ArrayVec)
        pdvArrayMat = term + term.T
        return pdvArrayMat

    def derArrayVec(self, arg, AntPos, az, el):
        """
        """
        WaveVec = self.WaveVec(az, el)
        pdvWaveVec = self.derWaveVec(arg, az, el)
        pdvArrayVec = 1j * np.dot(AntPos, pdvWaveVec) * \
                                  np.exp(1j * np.dot(AntPos, WaveVec))
        return pdvArrayVec

    def derWaveVec(self, arg, az, el):
        """
        """
        if arg == "azimuth":
            pdvWaveVec = 2*np.pi/self.lambda_ * \
                np.array([-np.sin(az)*np.sin(el), np.cos(az)*np.sin(el), 0])
        elif arg == "elevation":
            pdvWaveVec = 2*np.pi/self.lambda_ * \
                np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)])
        return pdvWaveVec

    def FIM_USU_core(self, alphal, taul, thetal, W, f):
        """
        Given channel parameters and the precoder and combiner, compute the
        Fisher information matrix for the USU signal.
        """
        L = len(alphal)
        delta_f = self.delta_f*1e-09

        # Prepare initial computations
        dn_taul = self.DelayVec(taul)
        aU_thetal = np.zeros((L, self.NU_prod), dtype=np.complex128)
        AU_ll = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        pdvAUaz = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        pdvAUel = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        for l in range(L):
            aU_thetal[l] = self.ArrayVec_(self.AntPos_UE, thetal[l, 0], thetal[l, 1])
            AU_ll[l] = np.outer(aU_thetal[l], aU_thetal[l])
            pdvAUaz[l] = self.derArrayMat("azimuth", self.AntPos_UE, thetal[l, 0], thetal[l, 1])
            pdvAUel[l] = self.derArrayMat("elevation", self.AntPos_UE, thetal[l, 0], thetal[l, 1])

        pdvRealAlpha = dn_taul[None, :, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", AU_ll, f))[:, None, :, :]
        pdvImagAlpha = 1j * pdvRealAlpha
        pdvDelay = -1j*2*np.pi*self.n[None, :, None, None]*delta_f*alphal[None, None, :, None] \
            * pdvRealAlpha
        pdvThetaAz = alphal[None, None, :, None] * dn_taul[None, :, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUaz, f))[:, None, :, :]
        pdvThetaEl = alphal[None, None, :, None] * dn_taul[None, :, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUel, f))[:, None, :, :]

        pdvVec = np.concatenate((pdvRealAlpha, pdvImagAlpha, pdvDelay, pdvThetaAz, pdvThetaEl), axis=2)
        FIM = 2*self.p_tx/self.p_noise * np.real(np.einsum("tnli,tnki->lk", pdvVec, pdvVec.conj()))
        return FIM

    def FIM_USU(self, Phi, sU, W, f):
        """
        From SP positions Phi and the UE state sU, compute the channel parameters.
        Then find the Fisher information matrix for the USU signal.
        """
        L, d = Phi.shape
        pU = sU[:d]

        taul, _ = self.nonRIS_delays(Phi, pU)
        alphal, _ = self.nonRIS_channel_coefficients(Phi, pU)
        self.alphal = alphal
        _, thetal = self.nonRIS_angles(Phi, sU)

        taul = taul*1e09
        return self.FIM_USU_core(alphal, taul, thetal, W, f)

    def FIM_USRU_core(self, alphal_bar, taul_bar, thetal, phil, theta0, phi0, W, omega, f):
        """
        """
        L = len(alphal_bar)
        delta_f = self.delta_f*1e-09

        # Prepare initial computations
        dn_taul_bar = self.DelayVec(taul_bar)
        aU_thetal = np.zeros((L, self.NU_prod), dtype=np.complex128)
        aU_theta0 = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        AU_0l = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        pdvAUaz = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        pdvAUel = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
        for l in range(L):
            aU_thetal[l] = self.ArrayVec_(self.AntPos_UE, thetal[l, 0], thetal[l, 1])
            AU_0l[l] = np.outer(aU_theta0, aU_thetal[l])
            pdvUEaz = self.derArrayVec("azimuth", self.AntPos_UE, thetal[l, 0], thetal[l, 1])
            pdvUEel = self.derArrayVec("elevation", self.AntPos_UE, thetal[l, 0], thetal[l, 1])
            pdvAUaz[l] = np.outer(aU_theta0, pdvUEaz)
            pdvAUel[l] = np.outer(aU_theta0, pdvUEel)

        aR_phi0 = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        aR_phil = np.zeros((L, self.NR_prod), dtype=np.complex128)
        pdvARaz = np.zeros((L, self.NR_prod), dtype=np.complex128)
        pdvARel = np.zeros((L, self.NR_prod), dtype=np.complex128)
        omegaProd = omega * aR_phi0[None, :]  # size=(T, NR)
        for l in range(L):
            aR_phil[l] = self.ArrayVec_(self.AntPos_RIS, phil[l, 0], phil[l, 1])
            pdvARaz[l] = self.derArrayVec("azimuth", self.AntPos_RIS, phil[l, 0], phil[l, 1])
            pdvARel[l] = self.derArrayVec("elevation", self.AntPos_RIS, phil[l, 0], phil[l, 1])
        RISterm = np.einsum("ti,li->tl", omegaProd, aR_phil)
        pdvRISaz = np.einsum("ti,li->tl", omegaProd, pdvARaz)
        pdvRISel = np.einsum("ti,li->tl", omegaProd, pdvARel)

        pdvRealAlpha = dn_taul_bar[None, :, :, None] * RISterm[:, None, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", AU_0l, f))[:, None, :, :]
        pdvImagAlpha = 1j * pdvRealAlpha
        pdvDelay = -1j*2*np.pi*self.n[None, :, None, None]*delta_f*alphal_bar[None, None, :, None] \
            * pdvRealAlpha
        pdvThetaAz = alphal_bar[None, None, :, None] * dn_taul_bar[None, :, :, None] * RISterm[:, None, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUaz, f))[:, None, :, :]
        pdvThetaEl = alphal_bar[None, None, :, None] * dn_taul_bar[None, :, :, None] * RISterm[:, None, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUel, f))[:, None, :, :]
        pdvPhiAz = alphal_bar[None, None, :, None] * dn_taul_bar[None, :, :, None] * pdvRISaz[:, None, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", AU_0l, f))[:, None, :, :]
        pdvPhiEl = alphal_bar[None, None, :, None] * dn_taul_bar[None, :, :, None] * pdvRISel[:, None, :, None] \
            * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", AU_0l, f))[:, None, :, :]

        pdvVec = np.concatenate((pdvRealAlpha, pdvImagAlpha, pdvThetaAz,
                                 pdvThetaEl, pdvDelay, pdvPhiAz, pdvPhiEl), axis=2)
        FIM = 2*self.p_tx/self.p_noise * np.real(np.einsum("tnli,tnki->lk", pdvVec, pdvVec.conj()))
        return FIM

    def FIM_USRU(self, Phi, sU, W, omega, f):
        """
        """
        L, d = Phi.shape
        pU = sU[:d]

        _, taul_bar, _ = self.RIS_delays(Phi, pU)
        _, alphal_bar, _ = self.RIS_channel_coefficients(Phi, pU)
        self.alphal_bar = alphal_bar
        theta0, thetal = self.nonRIS_angles(Phi, sU)
        phi0, phil = self.RIS_angles(Phi, pU)

        taul_bar = taul_bar*1e09
        return self.FIM_USRU_core(alphal_bar, taul_bar, thetal, phil, theta0, phi0, W, omega, f)

    def EFIM_solve(self, A, B, C):
        try:
            information_loss = np.dot(A, np.linalg.solve(B, C))
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                eps = 1e-01
                B = B*np.eye(B.shape[0])*eps
                if np.any(np.diag(B)) <= 0:
                    B = B + np.eye(B.shape[0])*eps
                information_loss = np.dot(A, np.linalg.solve(B, C))
            else:
                assert False
        return information_loss

    def EFIM(self, FIM, L, type_="nonRIS"):
        """
        """
        if type_ == "nonRIS":
            # information_loss = np.dot(FIM[2*L:, :2*L], np.linalg.solve(FIM[:2*L, :2*L], FIM[:2*L, 2*L:]))
            information_loss = self.EFIM_solve(FIM[2*L:, :2*L], FIM[:2*L, :2*L], FIM[:2*L, 2*L:])
            symmetric_information_loss = 1/2 * (information_loss + information_loss.T)
            EFIM = FIM[2*L:, 2*L:] - symmetric_information_loss
        elif type_ == "RIS":
            # information_loss = np.dot(FIM[4*L:, :4*L], np.linalg.solve(FIM[:4*L, :4*L], FIM[:4*L, 4*L:]))
            information_loss = self.EFIM_solve(FIM[4*L:, :4*L], FIM[:4*L, :4*L], FIM[:4*L, 4*L:])
            symmetric_information_loss = 1/2 * (information_loss + information_loss.T)
            EFIM = FIM[4*L:, 4*L:] - symmetric_information_loss
        if np.any(np.linalg.eigvals(EFIM) <= 0):
            EFIM *= np.eye(EFIM.shape[0])
        if np.any(np.diag(EFIM) <= 0):
            a = np.random.uniform(0.1, 2, size=(sum(np.diag(EFIM) <= 0)))
            EFIM_copy = np.copy(EFIM)
            np.fill_diagonal(EFIM, a)
            EFIM[EFIM_copy > 0] = EFIM_copy[EFIM_copy > 0]
        return EFIM

    def EFIMcombined(self, AlphaEst, AlphaBarEst, ChParsEst, WN, WR, f, omega, prior):
        """
        Find precision matrix as the equivalent Fisher information.
        """
        L_hat = AlphaEst.shape[0]

        FIM_USU = self.FIM_USU_core(AlphaEst, ChParsEst[:, 0], ChParsEst[:, 1:3], WN, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
        EFIM_USU = self.EFIM(FIM_USU, L_hat, type_="nonRIS")
        FIM_USRU = self.FIM_USRU_core(AlphaBarEst, ChParsEst[:, 3], ChParsEst[:, 1:3],
                                      ChParsEst[:, 4:], prior["theta0"], prior["phi0"], WR, omega, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
        EFIM_USRU = self.EFIM(FIM_USRU, L_hat, type_="RIS")
        EFIM_combined = np.zeros((6*L_hat, 6*L_hat))
        EFIM_combined[:3*L_hat, :3*L_hat] = EFIM_USU
        EFIM_combined[3*L_hat:, 3*L_hat:] = EFIM_USRU
        reEFIM_combined = self.rearrange_FIM(EFIM_combined, type_="Combined")
        return reEFIM_combined

    def CRLB(self, EFIM):
        """
        """
        # CRLB = np.linalg.inv(EFIM)
        CRLB = np.linalg.pinv(EFIM, hermitian=True)
        return CRLB

    def rearrange_FIM(self, EFIM, type_="nonRIS"):
        """
        Rearrange the rows and columns of the equivalent Fisher information
        matrix to give a precision matrix independently for each point.
        """
        if type_ == "nonRIS" or type_ == "RIS":
            dim = 3
        elif type_ == "Combined":
            dim = 6
        L = EFIM.shape[-1]//dim
        reEFIM = np.zeros((L*dim, L*dim))
        for l in range(L):
            idx_arr = np.array([j*L+l for j in range(dim)])
            reEFIM[l*dim:(l+1)*dim, l*dim:(l+1) * dim] = EFIM[idx_arr, :][:, idx_arr]
            for i in range(l):
                idx_arr2 = np.array([j*L+i for j in range(dim)])
                reEFIM[l*dim:(l+1)*dim, i*dim:(i+1) * dim] = EFIM[idx_arr, :][:, idx_arr2]
                reEFIM[i*dim:(i+1)*dim, l*dim:(l+1) * dim] = EFIM[idx_arr2, :][:, idx_arr]
        return reEFIM

    def Transformation(self, EFIM, Phi, sU, type_="nonRIS"):
        """
        """
        pU, oU = sU[:3], sU[3:]
        distance_diff = Phi - pU[None, :]  # size=(L, 3)

        if type_ == "nonRIS" or type_ == "Combined":
            RU = self.rot(*oU)
            p_SU = np.einsum("ij,li->lj", RU, distance_diff)  # size=(L, 3)
            p_SU_norm = np.linalg.norm(p_SU, axis=-1)

            dtheta_az = (RU[:, 1] * p_SU[:, 0, None] - RU[:, 0] * p_SU[:, 1, None]) \
                        / ((p_SU**2)[:, 0, None] + (p_SU**2)[:, 1, None])  # size=(L, 3)
            dtheta_el = -(RU[:, 2] * (p_SU_norm**2)[:, None] - p_SU[:, 2, None] * distance_diff) \
                    / ((p_SU_norm**3)[:, None] * np.sqrt(1 - (p_SU**2)[:, 2, None]/(p_SU_norm**2)[:, None]))  # size=(L, 3)
            dtau = 1/1e-09 * 2*distance_diff / (self.c*np.linalg.norm(distance_diff, axis=-1)[:, None])  # size=(L, 3)

        if type_ == "RIS" or type_ == "Combined":
            RIS_distance_diff = Phi - self.pR[None, :]  # size=(L, 3)
            RR = self.rot(*self.oR)  # size=(3, 3)
            p_SR = np.einsum("ij,ki->kj", RR, Phi -
                             self.pR[None, :])  # size=(L, 3)
            p_SR_norm = np.linalg.norm(p_SR, axis=1)

            dtau_bar = 1/1e-09 * RIS_distance_diff/(self.c*np.linalg.norm(RIS_distance_diff, axis=-1)[:, None]) \
                  + distance_diff / (self.c*np.linalg.norm(distance_diff, axis=-1)[:, None])  # size=(L, 3)
            dphi_az = (RR[None, :, 1] * p_SR[:, 0, None] - RR[None, :, 0] * p_SR[:, 1, None]) \
                        / ((p_SR**2)[:, 0, None] + (p_SR**2)[:, 1, None])  # size=(L, 3)
            dphi_el = -(RR[None, :, 2] * (p_SR_norm**2)[:, None] - p_SR[:, 2, None] * RIS_distance_diff) \
                    / ((p_SR_norm**3)[:, None] * np.sqrt(1 - (p_SR**2)[:, 2, None]/(p_SR_norm**2)[:, None]))  # size=(L, 3)

        if type_ == "nonRIS":
            deta = np.stack((dtau, dtheta_az, dtheta_el), axis=1)
        elif type_ == "RIS":
            deta = np.stack((dtau_bar, dphi_az, dphi_el), axis=1)
        elif type_ == "Combined":  # Combined
            deta = np.stack((dtau, dtheta_az, dtheta_el, dtau_bar, dphi_az, dphi_el), axis=1) # size=(L, 6, 3)

        T = block_diag(*deta) # size=(L*6, L*3)
        Jx = np.einsum("in,im->nm", T, np.einsum("ij,jm->im", EFIM, T)) # size=(L*3, L*3)
        return Jx

    def mainFisher(self, Phi, sU, alphal, taul, thetal, alphal_bar, taul_bar, phil,
                   theta0, phi0, WN, WR, omega, f):
        """
        Given inputs on the scenario, run the Fisher analysis
        for the USU and USRU signals as well as the combined. Save the resulting
        Cramér-Rao lower bounds in a dictionary, resFisher.
        """
        L = alphal.shape[0]

        FIM_USU = self.FIM_USU_core(alphal, taul, thetal, WN, f)
        EFIM_USU = self.EFIM(FIM_USU, L, type_="nonRIS")
        CRLB_USU = self.CRLB(EFIM_USU)
        reEFIM_USU = self.rearrange_FIM(EFIM_USU, type_="nonRIS")
        EFIMx_USU = self.Transformation(reEFIM_USU, Phi, sU, type_="nonRIS")
        CRLBx_USU = self.CRLB(EFIMx_USU)

        FIM_USRU = self.FIM_USRU_core(
            alphal_bar, taul_bar, thetal, phil, theta0, phi0, WR, omega, f)
        EFIM_USRU = self.EFIM(FIM_USRU, L, type_="RIS")
        CRLB_USRU = self.CRLB(EFIM_USRU)
        reEFIM_USRU = self.rearrange_FIM(EFIM_USRU, type_="RIS")
        EFIMx_USRU = self.Transformation(reEFIM_USRU, Phi, sU, type_="RIS")
        CRLBx_USRU = self.CRLB(EFIMx_USRU)

        EFIM_combined = np.zeros((6*L, 6*L))
        EFIM_combined[:3*L, :3*L] = EFIM_USU
        EFIM_combined[3*L:, 3*L:] = EFIM_USRU
        CRLB_combined = self.CRLB(EFIM_combined)
        reEFIM_combined = self.rearrange_FIM(EFIM_combined, type_="Combined")
        EFIMx_combined = self.Transformation(reEFIM_combined, Phi, sU, type_="Combined")
        CRLBx_combined = self.CRLB(EFIMx_combined)

        resFisher = {"CRLB_USU": CRLB_USU, "CRLBx_USU": CRLBx_USU,
                     "CRLB_USRU": CRLB_USRU, "CRLBx_USRU": CRLBx_USRU,
                     "CRLB_combined": CRLB_combined, "CRLBx_combined": CRLBx_combined}
        return resFisher

    def detection_probability_optimistic(self, Y, Phi, sU, W, f):
        """
        Compute the detection probability with the optimistic method.

        Note that with this method, the marginal detection probability
        increases as a weak target gets closer to a strong target which
        goes against intuition!
        """
        pU, _ = sU[:3], sU[3:]
        L = Phi.shape[0]
        T = Y.shape[0]
        Alpha = self.alphal

        # Compute channel parameters
        taul, _ = self.nonRIS_delays(Phi, pU)
        _, thetal = self.nonRIS_angles(Phi, sU)

        # Make design matrix
        Gmat = np.zeros((self.NU_prod*self.N*T, L), dtype=np.complex128)
        for l in range(L):
            gVec = np.sqrt(self.p_tx)*self.gN(taul[l], thetal[l], W, f, self.NU[0], self.NU[1], self.N, T)
            Gmat[:, l] = gVec

        # Compute matrix product and square root
        Gprod = np.dot(Gmat.conj().T, Gmat)
        sqrtGprod = sqrtm(Gprod)

        # Compute statistic
        GprodInv = np.linalg.inv(Gprod)
        GPseudoInverse = np.dot(GprodInv, Gmat.conj().T)
        AlphaEst = np.dot(GPseudoInverse, Y.flatten())

        gamma = np.sqrt(4/self.p_noise) * np.dot(sqrtGprod, AlphaEst)
        mu = np.zeros(L)
        for l in range(L):
            mu[l] = np.real(gamma[l])**2 + np.imag(gamma[l])**2

        # Make non-centrality parameters
        gamma_tilde = np.sqrt(4/self.p_noise) * np.dot(sqrtGprod, Alpha)
        mu_tilde = np.zeros(L)
        for l in range(L):
            mu_tilde[l] = np.real(gamma_tilde[l])**2 + np.imag(gamma_tilde[l])**2

        th_comp = 10000
        # gamma_th_arr = np.linspace(9.164e08, 9.17e08, th_sims)
        # gamma_th_arr = np.linspace(7.83e08, 7.88e08, th_sims)
        gamma_th_arr = np.linspace(1e-01, 3.5e5, th_comp)

        pD_marginal_arr = np.zeros((L, th_comp))
        for i, gamma_th in enumerate(gamma_th_arr):
            pD_marginal = np.zeros(L)
            for l in range(L):
                pD_marginal[l] = 1 - min(ncx2.cdf(gamma_th, 2, mu_tilde[l]), 1)
            pD_marginal_arr[:, i] = pD_marginal

            # if self.verboseEst is True:
            #     print("Marginal detection probabilities:\n", pD_marginal)

        # Compute false alarm probability
        pFA_marginal_arr = np.zeros((L, th_comp))
        for l in range(L):
            AlphaFA = np.copy(Alpha)
            AlphaFA[l] = 0
            gamma_tilde = np.sqrt(4/self.p_noise) * np.dot(sqrtGprod, AlphaFA)
            mu_tilde = np.real(gamma_tilde[l])**2 + np.imag(gamma_tilde[l])**2

            for i, gamma_th in enumerate(gamma_th_arr):
                pFA_marginal_arr[l, i] = 1 - min(ncx2.cdf(gamma_th, 2, mu_tilde), 1)

        plt.plot(gamma_th_arr, pD_marginal_arr[0, :], color="tab:orange", label="Target 1")
        plt.plot(gamma_th_arr, pD_marginal_arr[1, :], color="tab:green", label="Target 2")
        plt.xlabel("Threshold")
        plt.ylabel("Detection probability")
        plt.legend()
        plt.show()

        plt.plot(gamma_th_arr, pFA_marginal_arr[0, :], color="tab:orange", label="Target 1")
        plt.plot(gamma_th_arr, pFA_marginal_arr[1, :], color="tab:green", label="Target 2")
        plt.xlabel("Threshold")
        plt.ylabel("False alarm probability")
        plt.legend()
        plt.show()

        plt.plot(pFA_marginal_arr[0, :], pD_marginal_arr[0, :], color="tab:orange", label="Target 1")
        plt.plot(pFA_marginal_arr[1, :], pD_marginal_arr[1, :], color="tab:green", label="Target 2")
        plt.ylabel("Detection probability")
        plt.xlabel("False alarm probability")
        plt.legend()
        plt.show()
        return pD_marginal

    def detection_probability_greedy_core(self, Gmat, Alpha, sigma_alpha,
                greedy_method="orthogonal_matching_pursuit", estimation_method="least_squares", full_conditional="True"):
        """
        Compute the detection probability following a greedy algorithm.

        Inputs:
        -------
            Gmat : ndarray, size=(n, L)
            Alpha : ndarray, size=(L,)
            sigma_alpha : ndarray, size=(L,)
            greedy_method : str, options = "matching_pursuit", "orthogonal_matching_pursuit"
            estimation_method : str, options = "least_squares", "weighted_least_squares"
            full_conditional : str, options = "True", "False"

        Outputs:
        --------
            pFA : ndarray, size=(th_comp,)
            pD_arr : ndarray, size=(L, th_comp)
        """
        n, L = Gmat.shape
        th_comp = 100
        pFA = np.linspace(1e-09, 1, th_comp)
        gamma_th_arr = -2*np.log(pFA)

        if full_conditional == "False":
            # Simulate data for debugging
            if self.verboseEst is True:
                gamma = np.zeros((L), dtype=np.complex128)
                std = np.sqrt(self.p_noise/2)/np.sqrt(2)

                sims = 1000
                AlphaEst = np.zeros((sims, L, L), dtype=np.complex128)
                El = np.zeros((sims, L, L, n), dtype=np.complex128)
                gamma_tilde = np.zeros((sims, L), dtype=np.complex128)
                mu_tilde = np.zeros((sims, L))
                Ysim = np.zeros((sims, L, n), dtype=np.complex128)
                Alphasim = np.zeros((sims, L, L), dtype=np.complex128)
                for i in range(sims):
                    eps = np.random.normal(0, std, size=n) + 1j*np.random.normal(0, std, size=n)
                    nu = np.random.normal(0, np.sqrt(2)/2, size=L) + 1j*np.random.normal(0, np.sqrt(2)/2, size=L)
                    for j in range(L):
                        nu[:j+1] = np.ones(j+1)
                        Alphasim[i, j] = Alpha * nu
                        Ysim[i, j] = np.dot(Gmat, Alphasim[i, j]) + eps

            # Pre-compute outer products
            Gmat_prodl = np.zeros((L, n, n), dtype=np.complex128)
            for l in range(L):
                Gmat_prodl[l] = np.outer(Gmat[:, l], Gmat[:, l].conj().T)

            # Prepare loop
            pD_arr = np.zeros((L, th_comp))
            AlphaEstVar = np.zeros(L)
            Sigma_El = np.zeros((L, n, n), dtype=np.complex128)
            Sigma_El_inv = np.zeros((L, n, n), dtype=np.complex128)
            mul = np.zeros(L)
            for l in range(L):
                if greedy_method == "matching_pursuit":
                    Sigma_El[l] = self.p_noise/2 * np.eye(n)
                    for i in range(0, l):
                        Sigma_El[l] += AlphaEstVar[i] * Gmat_prodl[i]
                    for i in range(l+1, L):
                        Sigma_El[l] += sigma_alpha[i]**2 * Gmat_prodl[i]
                    Sigma_El_inv[l] = np.linalg.inv(Sigma_El[l])
    
                    if estimation_method == "least_squares":
                        AlphaEstVar[l] = np.real(np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El[l], Gmat[:, l]))) / np.linalg.norm(Gmat[:, l])**4
                    elif estimation_method == "weighted_least_squares":
                        AlphaEstVar[l] = 1/np.real(np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El_inv[l], Gmat[:, l])))

                    # Do estimation for debugging
                    if self.verboseEst is True:
                        for i in range(sims):
                            for j in range(L):
                                if l == 0:
                                    El[i, j, 0] = Ysim[i, j]
                                else:
                                    El[i, j, l] = El[i, j, l-1] - Gmat[:, l-1] * AlphaEst[i, j, l-1]
                                if estimation_method == "least_squares":
                                    AlphaEst[i, j, l] = np.dot(Gmat[:, l].conj().T, El[i, j, l]) / np.linalg.norm(Gmat[:, l])**2
                                elif estimation_method == "weighted_least_squares":
                                    AlphaEst[i, j, l] = np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El_inv[l], El[i, j, l])) / np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El_inv[l], Gmat[:, l]))
                                if j == l:
                                    gamma_tilde[i, l] = np.sqrt(2/AlphaEstVar[l]) * AlphaEst[i, j, l]
                                    mu_tilde[i, l] = 2/AlphaEstVar[l] * np.abs(AlphaEst[i, j, l])**2
                        gamma[l] = np.sqrt(2/AlphaEstVar[l]) * Alpha[l]

                mul[l] = 2/AlphaEstVar[l] * np.abs(Alpha[l])**2
                for i, gamma_th in enumerate(gamma_th_arr):
                    pD_arr[l, i] = 1 - min(ncx2.cdf(gamma_th, 2, mul[l]), 1)

        else:
            # assert L == 2, "Full conditional detection probability only implemented for L = 2!"
            assert greedy_method == "orthogonal_matching_pursuit" and estimation_method == "least_squares", "Limited implementation!"
            Alpha = Alpha*1e06
            p_noise = self.p_noise*1e12

            # Pre-compute cross-term outer products
            Gmat_prodkl = np.zeros((L, L, n, n), dtype=np.complex128)
            for k in range(L):
                for l in range(L):
                    Gmat_prodkl[k, l] = np.outer(Gmat[:, k], Gmat[:, l].conj().T)

            # Pre-compute norms
            Gmat_norml = np.linalg.norm(Gmat, axis=0)

            # Pre-compute projection matrices
            Proj_list = list()
            for l in range(L):
                Projl = np.dot(Gmat[:, 0:l], np.dot(np.linalg.inv(np.dot(Gmat[:, 0:l].conj().T, Gmat[:, 0:l])), Gmat[:, 0:l].conj().T))
                Proj_list.append(Projl)

            # Compute residual statistics
            Sigma_El_list = list()
            for l in range(L):
                Sigma_El = (np.eye(n) - Proj_list[l])
                Sigma_El_list.append(Sigma_El)

            # Compute bias terms
            bias_factorl_list = list()
            bias_factorl_list.append(np.dot(Gmat[:, 0].conj().T, np.dot(Gmat[:, 1:], Alpha[1:]))/Gmat_norml[0]**2)
            for l in range(1, L):
                bias_factorl = np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El_list[l], Gmat[:, l:]))/Gmat_norml[l]**2
                bias_factorl_list.append(bias_factorl)

            # Compute alpha statistics
            AlphaEstMeanl_list = list()
            AlphaEstVarl_list = list()
            AlphaEstMeanl_list.append(Alpha[0] + bias_factorl_list[0])
            AlphaEstVarl_list.append(p_noise/(2*Gmat_norml[0]**2))
            for l in range(1, L):
                AlphaEstMeanl = np.dot(bias_factorl_list[l], Alpha[l:])
                AlphaEstVarl = np.real(np.dot(Gmat[:, l].conj().T, np.dot(Sigma_El_list[l], Gmat[:, l])))/Gmat_norml[l]**4 * p_noise/2
                AlphaEstMeanl_list.append(AlphaEstMeanl)
                AlphaEstVarl_list.append(AlphaEstVarl)

            # Compute non-centrality parameters
            mul = np.zeros(L)
            for l in range(L):
                mul[l] = 2/AlphaEstVarl_list[l] * np.abs(AlphaEstMeanl_list[l])**2

            pD_arr = np.zeros((L, th_comp))
            for l in range(L):
                for i, gamma_th in enumerate(gamma_th_arr):
                    pD_arr[l, i] = 1 - min(ncx2.cdf(gamma_th, 2, mul[l]), 1)

        # Plotting for debugging
        if self.verboseEst is True:
            for l in range(L):
                # x_axis = np.linspace(np.min(np.real(gamma_tilde[:, l])), np.max(np.real(gamma_tilde[:, l])), 1000) 
                # plt.hist(np.real(gamma_tilde[:, l]), bins=50, density=True)
                # plt.plot(x_axis, norm.pdf(x_axis, np.real(gamma[l]), 1))
                # plt.show()
    
                # x_axis = np.linspace(np.min(np.imag(gamma_tilde[:, l])), np.max(np.imag(gamma_tilde[:, l])), 1000) 
                # plt.hist(np.imag(gamma_tilde[:, l]), bins=50, density=True)
                # plt.plot(x_axis, norm.pdf(x_axis, np.imag(gamma[l]), 1))
                # plt.show()

                # x_axis = np.linspace(np.min(mu_tilde[:, l]), np.max(mu_tilde[:, l]), 1000) 
                # plt.hist(mu_tilde[:, l], bins=50, density=True)
                # plt.plot(x_axis, ncx2.pdf(x_axis, 2, mul[l]))
                # plt.show()
    
                plt.plot(pFA, pD_arr[l], color="tab:green", label=f"Target {l+1}")
                plt.plot(pFA, pFA, color="black", linestyle="dashed")
                plt.ylabel("Detection probability")
                plt.xlabel("False alarm probability")
                plt.legend()
                plt.show()
        return pFA, pD_arr

    def detection_probability_greedy(self, YN, YR, Phi, sU, rcs, WN, WR, omega, f, greedy_method="orthogonal_matching_pursuit", estimation_method="least_squares", full_conditional="True"):
        """
        Compute the detection probability of each target following a
        greedy algorithm for both the non-RIS, RIS, and joint methods.

        Inputs:
        -------
            YN : ndarray, size=(T, N, NUx, NUy)
            YN : ndarray, size=(T, N, NUx, NUy)
            Phi : ndarray, size=(L, 3)
            sU : ndarray, size=(6,)
            rcs : ndarray, size=(L,)
            WN : ndarray, size=(Nrec, NUx * NUy)
            WR : ndarray, size=(Nrec, NUx * NUy)
            omega : ndarray, size=(T, NR)
            f : ndarray, size=(NUx * NUy)
            greedy_method : str, options = "matching_pursuit", "orthogonal_matching_pursuit"
            estimation_method : str, options = "least_squares", "weighted_least_squares"

        Outputs:
        -------
        
        """
        pU, _ = sU[:3], sU[3:]
        L = Phi.shape[0]
        T = YN.shape[0]

        ### Compute detection probability for non-RIS signal ###
        YN = YN.flatten()
        AlphaN = np.copy(self.alphal)

        # Compute channel parameters
        taul, _ = self.nonRIS_delays(Phi, pU)
        _, thetal = self.nonRIS_angles(Phi, sU)
        # sigma_alpha, _ = self.nonRIS_channel_gains(Phi, pU, rcs)
        # sigma_alpha = self.alphal_amp_mean
        sigma_alpha = np.abs(AlphaN)

        # Make design matrix
        GmatN = np.zeros((self.NU_prod*self.N*T, L), dtype=np.complex128)
        for l in range(L):
            gVec = np.sqrt(self.p_tx)*self.gN(taul[l], thetal[l], WN, f, self.NU[0], self.NU[1], self.N, T)
            GmatN[:, l] = gVec

        pFA, pD_arr_nonRIS = self.detection_probability_greedy_core(GmatN, AlphaN, sigma_alpha, greedy_method, estimation_method, full_conditional)

        ### Compute detection probability for RIS signal ###
        AlphaR = np.copy(self.alphal_bar)
        YR = YR.flatten()
        # print(np.abs(AlphaN), np.abs(AlphaR))

        # Compute channel parameters
        _, taul_bar, _ = self.RIS_delays(Phi, pU)
        phi0, phil = self.RIS_angles(Phi, pU)
        theta0, thetal = self.nonRIS_angles(Phi, sU)
        # _, sigma_alpha_bar, _ = self.RIS_channel_gains(Phi, pU, rcs)
        # sigma_alpha_bar = self.alphal_bar_amp_mean
        sigma_alpha_bar = np.abs(AlphaR)

        # Make design matrix
        GmatR = np.zeros((self.NU_prod*self.N*T, L), dtype=np.complex128)
        for l in range(L):
            gVec = np.sqrt(self.p_tx)*self.gR(taul_bar[l], phil[l], thetal[l], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
            GmatR[:, l] = gVec

        _, pD_arr_RIS = self.detection_probability_greedy_core(GmatR, AlphaR, sigma_alpha_bar, greedy_method, estimation_method, full_conditional)

        # pD_arr_nonRIS_intersection = np.prod(pD_arr_nonRIS, axis=0)
        # pD_arr_RIS_intersection = np.prod(pD_arr_RIS, axis=0)

        pD_arr_joint = pD_arr_nonRIS + pD_arr_RIS - pD_arr_nonRIS * pD_arr_RIS
        pFA_joint = 2*pFA - pFA**2
        # pD_arr_joint_intersection = pD_arr_nonRIS_intersection + pD_arr_RIS_intersection - pD_arr_nonRIS_intersection * pD_arr_RIS_intersection

        if self.verboseEst is True:
            for l in range(L):
                plt.title(f"Target {l+1}")
                plt.plot(pFA, pD_arr_nonRIS[l], color="tab:blue", label="non-RIS")
                plt.plot(pFA, pD_arr_RIS[l], color="tab:orange", label="RIS")
                plt.plot(pFA_joint, pD_arr_joint[l], color="tab:purple", label="Joint")
                plt.plot(pFA, pFA, color="black", linestyle="dashed")
                plt.ylabel("Detection probability")
                plt.xlabel("False alarm probability")
                plt.legend()
                plt.show()

                AUC_nonRIS = auc(pFA, pD_arr_nonRIS[l])
                AUC_RIS = auc(pFA, pD_arr_RIS[l])
                AUC_joint = auc(pFA_joint, pD_arr_joint[l])
                print(f"Area under the ROC curve, Target {l+1}:")
                print(f"Non-RIS : {AUC_nonRIS:.4f}\nRIS     : {AUC_RIS:.4f}\nJoint   : {AUC_joint:.4f}")

            # plt.title("All targets")
            # plt.plot(pFA, pD_arr_nonRIS_intersection, color="tab:blue", label="non-RIS")
            # plt.plot(pFA, pD_arr_RIS_intersection, color="tab:orange", label="RIS")
            # plt.plot(pFA_joint, pD_arr_joint_intersection, color="tab:purple", label="Joint")
            # plt.plot(pFA, pFA, color="black", linestyle="dashed")
            # plt.ylabel("Probability of correct detection")
            # plt.xlabel("False alarm probability")
            # plt.legend()
            # plt.show()

            for l in range(L):
                with open(f"results/pDgreedyTarget{l+1}.txt", "w") as file:
                    file.write("\\addplot[semithick, mark=star, mark options = {solid}, color2]\n")
                    file.write("table{%\n")
                    for x, y in zip(pFA, pD_arr_nonRIS[l]):
                        file.write(f"{x:.4f}  {y:.4f}\n")
                    file.write("}; \\addlegendentry{$p_{\\rm D, "+f"{l+1}"+"}^{\\rm N}$}\n")
                    file.write("\\addplot[semithick, mark=square, mark options = {solid}, color3]\n")
                    file.write("table{%\n")
                    for x, y in zip(pFA, pD_arr_RIS[l]):
                        file.write(f"{x:.4f}  {y:.4f}\n")
                    file.write("}; \\addlegendentry{$p_{\\rm D, "+f"{l+1}"+"}^{\\rm R}$}\n")
                    file.write("\\addplot[semithick, mark=diamond, mark options = {solid}, color4]\n")
                    file.write("table{%\n")
                    for x, y in zip(pFA, pD_arr_joint[l]):
                        file.write(f"{x:.4f}  {y:.4f}\n")
                    file.write("}; \\addlegendentry{$p_{\\rm D, "+f"{l+1}"+"}^{\\rm joint}$}\n")

            # with open("results/pDgreedyAll.txt", "w") as file:
            #     file.write("\\addplot[semithick, mark=star, mark options = {solid}, color2]\n")
            #     file.write("table{%\n")
            #     for x, y in zip(pFA, pD_arr_nonRIS_intersection):
            #         file.write(f"{x:.4f}  {y:.4f}\n")
            #     file.write("}; \\addlegendentry{$p_{\\rm D}^{\\rm N}$}\n")
            #     file.write("\\addplot[semithick, mark=square, mark options = {solid}, color3]\n")
            #     file.write("table{%\n")
            #     for x, y in zip(pFA, pD_arr_RIS_intersection):
            #         file.write(f"{x:.4f}  {y:.4f}\n")
            #     file.write("}; \\addlegendentry{$p_{\\rm D}^{\\rm R}$}\n")
            #     file.write("\\addplot[semithick, mark=diamond, mark options = {solid}, color4]\n")
            #     file.write("table{%\n")
            #     for x, y in zip(pFA, pD_arr_joint_intersection):
            #         file.write(f"{x:.4f}  {y:.4f}\n")
            #     file.write("}; \\addlegendentry{$p_{\\rm D}^{\\rm joint}$}")
        resDetect = {"pFA": pFA, "pD_arr_nonRIS": pD_arr_nonRIS, "pD_arr_RIS": pD_arr_RIS, "pD_arr_joint": pD_arr_joint}
        return resDetect

    # def mutual_coherence(self, tau, az, el, T, type_="nonRIS", theta0=None, phi0=None, omega=None):
    #     """
    #     Compute the maximum mutual coherence.

    #     Parameters
    #     ----------
    #     tau : ndarray, size=(K_delay,)
    #         The delays to search.
    #     az : ndarray, size=(K_az,)
    #         The azimuth angles to search.
    #     el : ndarray, size=(K_el,)
    #         The elevation angles to search.
    #     T : int
    #     type_ : str, optional
    #         The type of signal. Options are "nonRIS" or "RIS". The default is "nonRIS".
    #     theta0 : ndarray, size=(2,), optional
    #         Angle of arrival at the UE from the RIS. The default is None.
    #     phi0 : ndarray, size=(2,), optional
    #         Angle of arrival at the RIS from the UE. The default is None.
    #     omega : ndarray, size=(T, NR), optional
    #         The RIS phase profiles. The default is None.

    #     Returns
    #     -------
    #     max_mutual_coherence : float
    #     """
    #     K_delay, K_az, K_el = len(tau), len(az), len(el)
    #     M = K_delay*K_az*K_el
    #     Ny = T*self.N*self.NU_prod

    #     if type_ == "nonRIS":
    #         TimeVec = np.ones(T) # size=(T, )
    #         FreqVec = self.DelayVec(tau).T # size=(K_delay, N)
    #         AngleVec = self.ArrayVecTensor_(self.AntPos_UE, az, el) # size=(K_az, K_el, NU)
    #         g = np.einsum('t,daefx->daetfx', TimeVec, np.einsum('df,aex->daefx', FreqVec, AngleVec)).reshape((M, Ny)) # size=(M, Ny)
    #         mutual_coherence = np.abs(np.einsum("ij, kj->ik", g.conj(), g, optimize="greedy"))/Ny
    #         print(mutual_coherence)
    #     elif type_ == "RIS":
    #         aR = self.ArrayVecTensor_(self.AntPos_RIS, az, el) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])[None, None, :]  # (K_az, K_el, NR)
    #         TimeVec = np.einsum("ti, aei -> aet", omega, aR) # (K_az, K_el, T)
    #         FreqVec = self.DelayVec(tau).T # size=(K_delay, N)
    #         AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]) # size=(NU,)
    #         g = np.einsum('aet,dfx->daetfx', TimeVec, np.einsum('df,x->dfx', FreqVec, AngleVec)).reshape((M, Ny)) # size=(M, Ny)
    #         mutual_coherence = np.abs(np.einsum("ij, kj->ik", g.conj(), g, optimize="greedy"))/(Ny*self.NR_prod)
    #     mask = np.ones(mutual_coherence.shape, dtype=bool)
    #     np.fill_diagonal(mask, 0)
    #     max_mutual_coherence = mutual_coherence[mask].max()
    #     # max_mutual_coherence = np.max(mutual_coherence)
    #     return max_mutual_coherence

    def mutual_coherence(self, pars, T, type_="nonRIS", theta0=None, phi0=None, omega=None):
        """
        Compute the maximum mutual coherence.

        Parameters
        ----------
        pars : ndarray, size=(M,)
            The parameter combinations to search.
        T : int
        type_ : str, optional
            The type of signal. Options are "nonRIS" or "RIS". The default is "nonRIS".
        theta0 : ndarray, size=(2,), optional
            Angle of arrival at the UE from the RIS. The default is None.
        phi0 : ndarray, size=(2,), optional
            Angle of arrival at the RIS from the UE. The default is None.
        omega : ndarray, size=(T, NR), optional
            The RIS phase profiles. The default is None.

        Returns
        -------
        max_mutual_coherence : float
        """
        M = len(pars)
        Ny = T*self.N*self.NU_prod

        if type_ == "nonRIS":
            TimeVec = np.ones(T) # size=(T, )
            FreqVec = self.DelayVec(pars[:, 0]).T # size=(M, N)
            AngleVec = self.ArrayVec_(self.AntPos_UE, pars[:, 1], pars[:, 2]).T # size=(M, NU)
            g = np.einsum('t,mfx->mtfx', TimeVec, np.einsum('mf,mx->mfx', FreqVec, AngleVec)).reshape((M, Ny)) # size=(M, Ny)
            mutual_coherence = np.abs(np.einsum("ij, kj->ik", g.conj(), g, optimize="greedy"))/Ny
        elif type_ == "RIS":
            aR = self.ArrayVec_(self.AntPos_RIS, pars[:, 1], pars[:, 2]).T * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])[None, :]  # (M, NR)
            TimeVec = np.einsum("ti, mi -> mt", omega, aR) # (M, T)
            FreqVec = self.DelayVec(pars[:, 0]).T # size=(M, N)
            AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]) # size=(NU,)
            g = np.einsum('mt,mfx->mtfx', TimeVec, np.einsum('mf,x->mfx', FreqVec, AngleVec)).reshape((M, Ny)) # size=(M, Ny)
            mutual_coherence = np.abs(np.einsum("ij, kj->ik", g.conj(), g, optimize="greedy"))/(np.linalg.norm(g, axis=1)[:, None]*np.linalg.norm(g, axis=1)[None, :]) # (Ny*self.NR_prod)
        mask = np.ones(mutual_coherence.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_mutual_coherence = mutual_coherence[mask].max()
        # max_mutual_coherence = np.max(mutual_coherence)
        return max_mutual_coherence

    def detection_probability_lower_bound(self, parsN, parsR, omega, f, prior, hpars_grid):
        """
        Compute the lower bound of the detection probability with
        orthogonal matching pursuit based on the mutual coherence metric.

        Parameters
        ----------
        thetal : ndarray, size=(L, 2)
        omega : ndarray, size=(T, NR)
        f : ndarray, size=(NU,)
        prior : dict
        hpars_grid : dict

        Returns
        -------
        boundN : float
        boundR : float
        """
        T = omega.shape[0]
        Ny = T*self.N*self.NU_prod
        Erel = 1e-7 # Design parameter >= 0: if 0 false alarm is 100 %; if 0 detection probability is zero
        L = self.alphal.shape[0]
        sigma = np.sqrt(self.p_noise/2)

        ### Compute bound for non-RIS signal ###
        # delaysN = np.linspace(prior["tau_bounds"][0], prior["tau_bounds"][1], hpars_grid["K_delayN"])
        # az_anglesN = np.linspace(prior["theta_bounds"][0], prior["theta_bounds"][1], hpars_grid["K_azN"])
        # el_anglesN = np.linspace(prior["theta_bounds"][2], prior["theta_bounds"][3], hpars_grid["K_elN"])
        # MN = hpars_grid["K_delayN"]*hpars_grid["K_azN"]*hpars_grid["K_elN"]

        MN = L

        BetaN = np.copy(self.alphal) # size=(L,)
        a = np.zeros((L, self.NU_prod), dtype=np.complex128)
        for l in range(L):
            a[l] = self.ArrayVec_(self.AntPos_UE, parsN[l, 1], parsN[l, 2]) # size=(L, NU)
        inner_product_a_f = np.einsum("li,i->l", a, f) # size=(L,)
        AlphaN = BetaN * inner_product_a_f
        x_minN, x_maxN = np.min(np.abs(AlphaN)), np.max(np.abs(AlphaN))

        max_mutual_coherence_nonRIS = self.mutual_coherence(parsN, T, "nonRIS")

        rhoN = 1 - max_mutual_coherence_nonRIS*(L-1)
        DRN = x_maxN**2/x_minN**2
        SNRN = self.p_tx * x_minN**2/sigma**2
        # print(MN, Ny, rhoN, Erel, SNRN, max_mutual_coherence_nonRIS, L, DRN)

        P1N = MN/Ny * np.sqrt(2/(np.pi*rhoN**2*Erel*Ny*SNRN)) * np.exp(-Erel*rhoN**2*Ny*SNRN/2)
        P2N = 4*MN * np.exp(-1/(16*L**2*max_mutual_coherence_nonRIS**2*DRN/MN + 8*max_mutual_coherence_nonRIS*np.sqrt(DRN)/(3*np.sqrt(2))))
        boundN = max(0, (1 - P1N) * (1 - P2N))

        ### Compute bound for RIS signal ###
        # delaysR = np.linspace(prior["tau_bar_bounds"][0], prior["tau_bar_bounds"][1], hpars_grid["K_delayR"])
        # az_anglesR = np.linspace(prior["phi_bounds"][0], prior["phi_bounds"][1], hpars_grid["K_azR"])
        # el_anglesR = np.linspace(prior["phi_bounds"][2], prior["phi_bounds"][3], hpars_grid["K_elR"])
        # MR = hpars_grid["K_delayR"]*hpars_grid["K_azR"]*hpars_grid["K_elR"]

        MR = L

        BetaR = np.copy(self.alphal_bar) # size=(L,)
        AlphaR = BetaR * inner_product_a_f
        x_minR, x_maxR = np.min(np.abs(AlphaR)), np.max(np.abs(AlphaR))

        max_mutual_coherence_RIS = self.mutual_coherence(parsR, T, "RIS", prior["theta0"], prior["phi0"], omega)

        rhoR = 1 - max_mutual_coherence_RIS*(L-1)
        DRR = x_maxR**2/x_minR**2
        SNRR = self.p_tx * x_minR**2/sigma**2
        Ny = (T*self.N*self.NU_prod*self.NR_prod).astype(np.int64)*self.NR_prod

        # print(MR, Ny, rhoR, Erel, SNRR, max_mutual_coherence_RIS, L, DRR)
        P1R = MR/Ny * np.sqrt(2/(np.pi*rhoR**2*Erel*Ny*SNRR)) * np.exp(-Erel*rhoR**2*Ny*SNRR/2)
        P2R = 4*MR * np.exp(-1/(16*L**2*max_mutual_coherence_RIS**2*DRR/MR + 8*max_mutual_coherence_RIS*np.sqrt(DRR)/(3*np.sqrt(2))))
        boundR = max(0, (1 - P1R) * (1 - P2R))
        return boundN, boundR

    def RunChAnalysis(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        Analyze theoretical properties of high-resolution sensing algorithm.
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)
        L = Phi.shape[0]

        # Simulate signal model
        YN, YR, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]
        YN, YR = YN.reshape((self.T2//2, self.N, self.NU[0], self.NU[1])), YR.reshape((self.T2//2, self.N, self.NU[0], self.NU[1]))

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        if L > 0:
            if run_fisher is True:
                resFisher = self.mainFisher(Phi, sU, self.alphal, ChPars[:, 0], ChPars[:, 1:3], self.alphal_bar,
                                            ChPars[:, 3], ChPars[:, 4:], prior["theta0"], prior["phi0"], WN, WR,
                                            omega, np.repeat(np.expand_dims(f, axis=0), self.T2//2, axis=0))
            else:
                resFisher = dict()
            if run_detect is True:
                resDetect = self.detection_probability_greedy(YN, YR, Phi, sU, rcs, WN, WR, omega, f)
                # hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                #               "K_delayR": hpars["K_delayR"], "K_azR": hpars["K_azR"], "K_elR": hpars["K_elR"]}
                # parsN = np.concatenate((np.expand_dims(ChPars[:, 0], axis=1)*1e-09, ChPars[:, 1:3]), axis=1)
                # parsR = np.concatenate((np.expand_dims(ChPars[:, 3], axis=1)*1e-09, ChPars[:, 4:]), axis=1)
                # boundN, boundR = self.detection_probability_lower_bound(parsN, parsR, omega, f, prior, hpars_grid)
                # resDetect = {"boundN": boundN, "boundR": boundR}
                # print(resDetect)
            else:
                resDetect = dict()
        else:
            resFisher = dict()
            resDetect = dict()
        return resDetect, resFisher

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
    mod = ChannelAnalysis(None, **toml_settings)
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
    # Run channel analysis
    # =============================================================================
    # bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
    #           "theta_bounds": np.array([0.57, 0.87, 0.65, 0.95]),
    #           "tau_bar_bounds": np.array([1.13e-07, 1.32e-07]),
    #           "phi_bounds": np.array([0.10, 0.78, 0.55, 1.16])}
    bounds = {"tau_bounds": np.array([1.12e-07, 1.28e-07]),
              "theta_bounds": np.array([0.55, 0.85, 0.65, 0.95])}
    toml_estimation["simulate_prior"] = False

    mod.RunChAnalysis(Phi, sU, rcs, toml_estimation, True, bounds)

    # with open("results/temp.pickle", "wb") as file:
    #     pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

    # mod.run_profile()

if __name__ == "__main__":
    main()
