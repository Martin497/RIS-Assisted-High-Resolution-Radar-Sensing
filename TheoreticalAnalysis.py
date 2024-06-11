# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:44:48 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Various options for theoretical insights: coherence, TildeC,
           detection probability special case, expected detection probability,
           expected AUC, Fisher information special case.
"""

import numpy as np

from ChAnalysis import ChannelAnalysis


class TheoreticalInsights(ChannelAnalysis):
    """
    """
    def __init__(self, config_file, **kwargs):
        """
        """
        super(TheoreticalInsights, self).__init__(config_file, **kwargs)

    # def FIM_USU_partial(self, alphal, taul, thetal, W, f):
    #     """
    #     """
    #     T = self.T2//2

    #     g1n = self.gN(taul[0], thetal[0], W, f, self.NU[0], self.NU[1], self.N, T)
    #     g2n = self.gN(taul[1], thetal[1], W, f, self.NU[0], self.NU[1], self.N, T)

    #     # gNtau1 = self.dergNtau(taul[0], thetal[0], f)
    #     # gNtau2 = self.dergNtau(taul[1], thetal[1], f)

    #     gNthetaAz1 = self.dergNthetaAz(taul[0], thetal[0], f)
    #     gNthetaAz2 = self.dergNthetaAz(taul[1], thetal[1], f)

    #     # gNthetaEl1 = self.dergNthetaEl(taul[0], thetal[0], f)
    #     # gNthetaEl2 = self.dergNthetaEl(taul[1], thetal[1], f)

    #     # pdvVec1 = np.hstack((g1n, 1j*g1n, -1j*2*np.pi*self.delta_f*alphal[0]*gNtau1, alphal[0]*gNthetaAz1, alphal[0]*gNthetaEl1))
    #     # pdvVec2 = np.hstack((g2n, 1j*g2n, -1j*2*np.pi*self.delta_f*alphal[1]*gNtau2, alphal[1]*gNthetaAz2, alphal[1]*gNthetaEl2))

    #     pdvVec = np.stack((g1n, g2n, 1j*g1n, 1j*g2n, alphal[0]*gNthetaAz1, alphal[1]*gNthetaAz2))
    #     FIM = 2*self.p_tx/self.p_noise * np.real(np.dot(pdvVec, pdvVec.conj().T))
    #     return FIM

    # def FIM_USU_partial(self, alphal, taul, thetal, W, f):
    #     """
    #     Given channel parameters and the precoder and combiner, compute the
    #     Fisher information matrix for the USU signal.
    #     """
    #     L = len(alphal)
    #     delta_f = self.delta_f*1e-09

    #     # Prepare initial computations
    #     dn_taul = self.DelayVec(taul)
    #     aU_thetal = np.zeros((L, self.NU_prod), dtype=np.complex128)
    #     AU_ll = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
    #     pdvAUaz = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
    #     pdvAUel = np.zeros((L, self.NU_prod, self.NU_prod), dtype=np.complex128)
    #     for l in range(L):
    #         aU_thetal[l] = self.ArrayVec_(self.AntPos_UE, thetal[l, 0], thetal[l, 1])
    #         AU_ll[l] = np.outer(aU_thetal[l], aU_thetal[l])
    #         pdvAUaz[l] = self.derArrayMat("azimuth", self.AntPos_UE, thetal[l, 0], thetal[l, 1])
    #         pdvAUel[l] = self.derArrayMat("elevation", self.AntPos_UE, thetal[l, 0], thetal[l, 1])

    #     pdvRealAlpha = dn_taul[None, :, :, None] \
    #         * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", AU_ll, f))[:, None, :, :]
    #     pdvImagAlpha = 1j * pdvRealAlpha
    #     pdvDelay = -1j*2*np.pi*self.n[None, :, None, None]*delta_f*alphal[None, None, :, None] \
    #         * pdvRealAlpha
    #     pdvThetaAz = alphal[None, None, :, None] * dn_taul[None, :, :, None] \
    #         * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUaz, f))[:, None, :, :]
    #     pdvThetaEl = alphal[None, None, :, None] * dn_taul[None, :, :, None] \
    #         * np.einsum("ij,tli->tlj", W.conj(), np.einsum("lij,tj->tli", pdvAUel, f))[:, None, :, :]

    #     pdvVec = np.concatenate((pdvThetaAz, pdvRealAlpha, pdvImagAlpha, pdvThetaEl, pdvDelay), axis=2)
    #     FIM = 2*self.p_tx/self.p_noise * np.real(np.einsum("tnli,tnki->lk", pdvVec, pdvVec.conj()))
    #     return FIM

    # def ComputeFisherPartial(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
    #     """
    #     """
    #     self.verboseEst = verboseEst
    #     self.bounds = bounds

    #     Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

    #     # Simulate signal model
    #     _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)

    #     if self.verboseEst is True:
    #         if Phi.shape[0] > 0:
    #             print("Channel coefficients SB: \n", self.alphal)
    #             print("Channel coefficients DB: \n", self.alphal_bar)

    #     FIM = self.FIM_USU_partial(self.alphal, ChPars[:, 0], ChPars[:, 1:3], WN, f)
    #     information_loss = np.dot(FIM[0, 1:], np.linalg.solve(FIM[1:, 1:], FIM[1:, 0]))
    #     symmetric_information_loss = 1/2 * (information_loss + information_loss.T)
    #     EFIMthetaAz1 = FIM[0, 0] - symmetric_information_loss
    #     return EFIMthetaAz1

    def dergNtau(self, tau, theta, f):
        """
        """
        aU = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1])
        TimeVec = np.ones(self.T2//2)
        FreqVec = self.DelayVec(tau).flatten() * np.arange(self.N)
        dergN = np.dot(aU, f) * np.kron(TimeVec, np.kron(FreqVec, aU))
        return dergN

    def dergNthetaAz(self, tau, theta, f):
        """
        """
        deraU = self.derArrayVec("azimuth", self.AntPos_UE, theta[0], theta[1])
        aU = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1])
        inner_prod = np.dot(aU, f)
        der_inner_prod = np.dot(deraU, f)
        TimeVec = np.ones(self.T2//2)
        FreqVec = self.DelayVec(tau).flatten()
        AngleVec = deraU * inner_prod + aU * der_inner_prod
        dergN = np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergN

    def dergNthetaEl(self, tau, theta, f):
        """
        """
        deraU = self.derArrayVec("elevation", self.AntPos_UE, theta[0], theta[1])
        aU = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1])
        inner_prod = np.dot(aU, f)
        der_inner_prod = np.dot(deraU, f)
        TimeVec = np.ones(self.T2//2)
        FreqVec = self.DelayVec(tau).flatten()
        AngleVec = deraU * inner_prod + aU * der_inner_prod
        dergN = np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergN

    def dergRtau(self, theta, tau_bar, phi, phi0, theta0, omega, f):
        """
        """
        aU0 = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        aU = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1])
        aR = self.ArrayVec_(self.AntPos_RIS, phi[0], phi[1]) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        TimeVec = np.dot(omega, aR)
        FreqVec = self.DelayVec(tau_bar).flatten() * np.arange(self.N)
        dergR = np.dot(aU, f) * np.kron(TimeVec, np.kron(FreqVec, aU0))
        return dergR

    def dergRthetaAz(self, theta, tau_bar, phi, phi0, theta0, omega, f):
        """
        """
        inner_prod = np.dot(self.derArrayVec("azimuth", self.AntPos_UE, theta[0], theta[1]), f)
        aR = self.ArrayVec_(self.AntPos_RIS, phi[0], phi[1]) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        TimeVec = np.dot(omega, aR)
        FreqVec = self.DelayVec(tau_bar).flatten()
        AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        dergR = inner_prod * np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergR

    def dergRthetaEl(self, theta, tau_bar, phi, phi0, theta0, omega, f):
        """
        """
        inner_prod = np.dot(self.derArrayVec("elevation", self.AntPos_UE, theta[0], theta[1]), f)
        aR = self.ArrayVec_(self.AntPos_RIS, phi[0], phi[1]) * self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        TimeVec = np.dot(omega, aR)
        FreqVec = self.DelayVec(tau_bar).flatten()
        AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        dergR = inner_prod * np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergR

    def dergRphiAz(self, theta, tau_bar, phi, phi0, theta0, omega, f):
        """
        """
        inner_prod = np.dot(self.ArrayVec_(self.AntPos_UE, theta[0], theta[1]), f)
        aR = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1]) * self.derArrayVec("azimuth", self.AntPos_RIS, phi[0], phi[1])
        TimeVec = np.dot(omega, aR)
        FreqVec = self.DelayVec(tau_bar).flatten()
        AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        dergR = inner_prod * np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergR

    def dergRphiEl(self, theta, tau_bar, phi, phi0, theta0, omega, f):
        """
        """
        inner_prod = np.dot(self.ArrayVec_(self.AntPos_UE, theta[0], theta[1]), f)
        aR = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1]) * self.derArrayVec("elevation", self.AntPos_RIS, phi[0], phi[1])
        TimeVec = np.dot(omega, aR)
        FreqVec = self.DelayVec(tau_bar).flatten()
        AngleVec = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1])
        dergR = inner_prod * np.kron(TimeVec, np.kron(FreqVec, AngleVec))
        return dergR

    def ArrayResponseInnerProducts(self, taul, thetal, taul_bar, phil, phi0, theta0, WN, WR, omega, f):
        """
        """
        aU1 = self.ArrayVec_(self.AntPos_UE, thetal[0, 0], thetal[0, 1])
        aU2 = self.ArrayVec_(self.AntPos_UE, thetal[1, 0], thetal[1, 1])
        deraU1 = self.derArrayVec("azimuth", self.AntPos_UE, thetal[0, 0], thetal[0, 1])
        deraU2 = self.derArrayVec("azimuth", self.AntPos_UE, thetal[1, 0], thetal[1, 1])

        # aU1aU2 = np.dot(aU1.conj(), aU2)
        # aU1deraU2 = np.dot(aU1.conj(), deraU2)
        # aU2deraU1 = np.dot(aU2.conj(), deraU1)
        # deraU1deraU2 = np.dot(deraU1.conj(), deraU2)

        aR0 = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1])
        aR1 = self.ArrayVec_(self.AntPos_RIS, phil[0, 0], phil[0, 1])
        aR2 = self.ArrayVec_(self.AntPos_RIS, phil[1, 0], phil[1, 1])
        deraR1 = self.derArrayVec("azimuth", self.AntPos_RIS, phil[0, 0], phil[0, 1])
        deraR2 = self.derArrayVec("azimuth", self.AntPos_RIS, phil[1, 0], phil[1, 1])

        nu1 = np.dot(omega * aR0, aR1)
        nu2 = np.dot(omega * aR0, aR2)
        dernu1 = np.dot(omega * aR0, deraR1)
        dernu2 = np.dot(omega * aR0, deraR2)
        return aU1, aU2, deraU1, deraU2, nu1, nu2, dernu1, dernu2

    def ComputeArrayResponseInnerProducts(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        aU1, aU2, deraU1, deraU2, nu1, nu2, dernu1, dernu2 \
            = self.ArrayResponseInnerProducts(ChPars[:, 0]*1e-09, ChPars[:, 1:3], ChPars[:, 3]*1e-09, ChPars[:, 4:],
                                              prior["phi0"], prior["theta0"], WN, WR, omega, f)
        return aU1, aU2, deraU1, deraU2, nu1, nu2, dernu1, dernu2


    def FisherTerms(self, g1, g2, gthetaAz1, gthetaAz2):
        """
        """
        # g inner products
        der1der2 = np.dot(gthetaAz1.conj(), gthetaAz2)
        g1der1 = np.dot(g1.conj(), gthetaAz1)
        g1der2 = np.dot(g1.conj(), gthetaAz2)
        g2der1 = np.dot(g2.conj(), gthetaAz1)
        g2der2 = np.dot(g2.conj(), gthetaAz2)
        g1g2 = np.dot(g1.conj(), g2)

        # g norms
        g1norm = np.linalg.norm(g1)
        g2norm = np.linalg.norm(g2)
        gthetaAz1norm = np.linalg.norm(gthetaAz1)
        gthetaAz2norm = np.linalg.norm(gthetaAz2)

        # g normalized coherence
        muC = np.abs(g1g2)**2/(g1norm**2 * g2norm**2)

        # x vectors
        xthetaAz1 = g2 * g1der1 - g1 * g2der1
        xthetaAz2 = g2 * g1der2 - g1 * g2der2

        # x inner product
        xprod = np.dot(xthetaAz1.conj(), xthetaAz2)

        # x norm
        xthetaAz1norm = np.linalg.norm(xthetaAz1)
        xthetaAz2norm = np.linalg.norm(xthetaAz2)

        term1 = gthetaAz1norm**2 - xthetaAz1norm**2/(g1norm**2 * g2norm**2 * (1 - muC))
        term2 = gthetaAz2norm**2 - xthetaAz2norm**2/(g1norm**2 * g2norm**2 * (1 - muC))
        term3 = np.real(der1der2)**2 + np.real(xprod)**2/(g1norm**2 * g2norm**2 * (1 - muC))**2 - 2/(g1norm**2 * g2norm**2 * (1 - muC)) * np.real(der1der2)*np.real(xprod)
        term4 = np.imag(der1der2)**2 + np.imag(xprod)**2/(g1norm**2 * g2norm**2 * (1 - muC))**2 - 2/(g1norm**2 * g2norm**2 * (1 - muC)) * np.imag(der1der2)*np.imag(xprod)
        term5 = np.real(der1der2)*np.imag(der1der2) + np.real(xprod)*np.imag(xprod)/(g1norm**2 * g2norm**2 * (1 - muC))**2 - (np.real(der1der2)*np.imag(xprod) + np.real(xprod)*np.imag(der1der2))/(g1norm**2 * g2norm**2 * (1 - muC))
        return term1, term2, term3, term4, term5, muC

    def ComputeFisherTerms(self, tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f):
        """
        """
        T = omega.shape[0]

        gN1 = self.gN(tau[0], theta[0], WN, f, self.NU[0], self.NU[1], self.N, T)
        gN2 = self.gN(tau[1], theta[1], WN, f, self.NU[0], self.NU[1], self.N, T)
        gNthetaAz1 = self.dergNthetaAz(tau[0], theta[0], f)
        gNthetaAz2 = self.dergNthetaAz(tau[1], theta[1], f)
        term1N, term2N, term3N, term4N, term5N, muCN = self.FisherTerms(gN1, gN2, gNthetaAz1, gNthetaAz2)

        gR1 = self.gR(tau_bar[0], phi[0], theta[0], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        gR2 = self.gR(tau_bar[1], phi[1], theta[1], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        gRphiAz1 = self.dergRphiAz(theta[0], tau_bar[0], phi[0], phi0, theta0, omega, f)
        gRphiAz2 = self.dergRphiAz(theta[1], tau_bar[1], phi[1], phi0, theta0, omega, f)
        term1R, term2R, term3R, term4R, term5R, muCR = self.FisherTerms(gR1, gR2, gRphiAz1, gRphiAz2)

        return term1N, term2N, term3N, term4N, term5N, muCN, term1R, term2R, term3R, term4R, term5R, muCR

    def SimplifiedFisherAz(self, alpha1, alpha2, term1, term2, term3, term4, term5):
        """
        """
        information = np.abs(alpha1)**2 * term1
        il_num = np.real(np.conj(alpha1)*alpha2)**2 * term3 + np.imag(np.conj(alpha1)*alpha2)**2 * term4 - 2*np.real(np.conj(alpha1)*alpha2)*np.imag(np.conj(alpha1)*alpha2) * term5
        il_denum = np.abs(alpha2)**2 * term2
        return np.where(information - il_num/il_denum >= 0, information - il_num/il_denum, 0)

    def ComputeFisherInnerProducts(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        term1N, term2N, term3N, term4N, term5N, muCN, term1R, term2R, term3R, term4R, term5R, muCR \
            = self.ComputeFisherTerms(ChPars[:, 0]*1e-09, ChPars[:, 1:3], ChPars[:, 3]*1e-09, ChPars[:, 4:],
                                      prior["phi0"], prior["theta0"], WN, WR, omega, f)
        return term1N, term2N, term3N, term4N, term5N, muCN, term1R, term2R, term3R, term4R, term5R, muCR

    def NormalizedCoherence(self, tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f):
        """
        """
        T = omega.shape[0]

        g1n = self.gN(tau[0], theta[0], WN, f, self.NU[0], self.NU[1], self.N, T)
        g2n = self.gN(tau[1], theta[1], WN, f, self.NU[0], self.NU[1], self.N, T)
        coherenceNormalizedN = np.abs(np.dot(g1n.conj(), g2n))**2/(np.linalg.norm(g1n)**2 * np.linalg.norm(g2n)**2)
        # varN = (np.linalg.norm(g2n)**2 - np.abs(np.dot(g1n.conj(), g2n))**2/np.linalg.norm(g1n)**2)/(2*np.linalg.norm(g2n)**4)

        g1r = self.gR(tau_bar[0], phi[0], theta[0], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        g2r = self.gR(tau_bar[1], phi[1], theta[1], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)

        coherenceNormalizedR = np.abs(np.dot(g1r.conj(), g2r))**2/(np.linalg.norm(g1r)**2 * np.linalg.norm(g2r)**2)
        # varR = (np.linalg.norm(g2r)**2 - np.abs(np.dot(g1r.conj(), g2r))**2/np.linalg.norm(g1r)**2)/(2*np.linalg.norm(g2r)**4)
        return coherenceNormalizedN, coherenceNormalizedR

    def ComputeNormalizedCoherence(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, rcs, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        # varphi = np.load(f"optimized_RIS_phase_profiles/varphi_T{self.T2//2}_NR{self.NR[0]}.npy")
        # omega = np.exp(1j*varphi)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        coherenceNormalizedN, coherenceNormalizedR \
            = self.NormalizedCoherence(ChPars[:, 0]*1e-09, ChPars[:, 1:3], ChPars[:, 3]*1e-09, ChPars[:, 4:],
                                       prior["phi0"], prior["theta0"], WN, WR, omega, f)
        return coherenceNormalizedN, coherenceNormalizedR

    def ExpectedDetectionProbability(self, pFA, rcs, P, g1, g2, g3):
        """
        """
        gamma_th = -2*np.log(pFA)

        if g3 is not None:
            g1norm = np.linalg.norm(g1)
            g2norm = np.linalg.norm(g2)
            g3norm = np.linalg.norm(g3)

            g1g2 = np.dot(g1.conj(), g2)
            g1g3 = np.dot(g1.conj(), g3)
            g2g3 = np.dot(g2.conj(), g3)

            C12 = np.abs(g1g2)**2/(g1norm**2 * g2norm**2)
            C13 = np.abs(g1g3)**2/(g1norm**2 * g3norm**2)
            C23 = np.abs(g2g3)**2/(g2norm**2 * g3norm**2)

            pd1_sum = np.sum([power**2*gnorm**2*C1l for power, gnorm, C1l in zip(P, [g1norm, g2norm, g3norm], [1, C12, C13])])
            BreveC = (C12*C13 + C23)/(1 - C12) - (2*np.real(g1g2*g1g3.conj()*g2g3))/(g1norm**2 * g2norm**2 * g3norm**2 * (1 - C12))
            TildeC = (C13 + C23)/(1 - C12) - (2*np.real(g1g2*g1g3.conj()*g2g3))/(g1norm**2 * g2norm**2 * g3norm**2 * (1 - C12))

            pd1 = np.exp(-(np.pi*self.p_noise*gamma_th)/(16*pd1_sum + 2*np.pi*self.p_noise))
            pd2 = np.exp(-(np.pi*self.p_noise*gamma_th)/(16*(P[1]**2*g2norm**2*(1-C12) + P[2]**2*g3norm**2*BreveC) + 2*np.pi*self.p_noise))
            pd3 = np.exp(-(np.pi*self.p_noise*gamma_th)/(16*P[2]**2*g3norm**2*(1-TildeC) + 2*np.pi*self.p_noise))
        else:
            g1norm = np.linalg.norm(g1)
            g2norm = np.linalg.norm(g2)

            g1g2 = np.dot(g1.conj(), g2)

            C12 = np.abs(g1g2)**2/(g1norm**2 * g2norm**2)

            pd1_sum = np.sum([power**2*gnorm**2*C1l for power, gnorm, C1l in zip(P, [g1norm, g2norm], [1, C12])])

            pd1 = np.exp(-(np.pi*self.p_noise*gamma_th)/(16*pd1_sum + 2*np.pi*self.p_noise))
            pd2 = np.exp(-(np.pi*self.p_noise*gamma_th)/(16*(P[1]**2*g2norm**2*(1-C12)) + 2*np.pi*self.p_noise))
            pd3 = None
        return pd1, pd2, pd3

    def ExpectedDetectionProbabilityWrapper(self, pFA, rcs, P, tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f):
        """
        """
        T = omega.shape[0]
        g1n = self.gN(tau[0], theta[0], WN, f, self.NU[0], self.NU[1], self.N, T)
        g2n = self.gN(tau[1], theta[1], WN, f, self.NU[0], self.NU[1], self.N, T)
        if len(tau) == 3:
            g3n = self.gN(tau[2], theta[2], WN, f, self.NU[0], self.NU[1], self.N, T)
            pd1n, pd2n, pd3n = self.ExpectedDetectionProbability(pFA, rcs, self.Palphal, g1n, g2n, g3n)
        else:
            pd1n, pd2n, pd3n = self.ExpectedDetectionProbability(pFA, rcs, self.Palphal, g1n, g2n, None)

        g1r = self.gR(tau_bar[0], phi[0], theta[0], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        g2r = self.gR(tau_bar[1], phi[1], theta[1], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        if len(tau_bar) == 3:
            g3r = self.gR(tau_bar[2], phi[2], theta[2], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
            pd1r, pd2r, pd3r = self.ExpectedDetectionProbability(pFA, rcs, self.Palphal_bar, g1r, g2r, g3r)
        else:
            pd1r, pd2r, pd3r = self.ExpectedDetectionProbability(pFA, rcs, self.Palphal_bar, g1r, g2r, None)
        return pd1n, pd2n, pd3n, pd1r, pd2r, pd3r

    def ComputeExpectedDetectionProbability(self, pFA, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, _, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        pd1n, pd2n, pd3n, pd1r, pd2r, pd3r \
            = self.ExpectedDetectionProbabilityWrapper(pFA, rcs, self.Palphal, ChPars[:, 0]*1e-09, ChPars[:, 1:3], ChPars[:, 3]*1e-09, ChPars[:, 4:],
                                                       prior["phi0"], prior["theta0"], WN, WR, omega, f)
        return pd1n, pd2n, pd3n, pd1r, pd2r, pd3r


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
        parsN : ndarray, size=(L, 3)
        parsR : ndarray, size=(L, 3)
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
        boundN = max(0, 1 - P1N) * max(0, 1 - P2N)

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

    def ComputeDetectionProbabilityLowerBound(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, _, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        hpars_grid = {"K_delayN": hpars["K_delayN"], "K_azN": hpars["K_azN"], "K_elN": hpars["K_elN"],
                      "K_delayR": hpars["K_delayR"], "K_azR": hpars["K_azR"], "K_elR": hpars["K_elR"]}
        parsN = np.concatenate((np.expand_dims(ChPars[:, 0], axis=1)*1e-09, ChPars[:, 1:3]), axis=1)
        parsR = np.concatenate((np.expand_dims(ChPars[:, 3], axis=1)*1e-09, ChPars[:, 4:]), axis=1)
        pdn, pdr = self.detection_probability_lower_bound(parsN, parsR, omega, f, prior, hpars_grid)
        return pdn, pdr

    def TildeC(self, g1, g2, g3):
        """
        """
        g1norm = np.linalg.norm(g1)
        g2norm = np.linalg.norm(g2)
        g3norm = np.linalg.norm(g3)

        g1g2 = np.dot(g1.conj(), g2)
        g1g3 = np.dot(g1.conj(), g3)
        g2g3 = np.dot(g2.conj(), g3)

        C12 = np.abs(g1g2)**2/(g1norm**2 * g2norm**2)
        C13 = np.abs(g1g3)**2/(g1norm**2 * g3norm**2)
        C23 = np.abs(g2g3)**2/(g2norm**2 * g3norm**2)

        TildeC = (C13 + C23)/(1 - C12) - (2*np.real(g1g2*g1g3.conj()*g2g3))/(g1norm**2 * g2norm**2 * g3norm**2 * (1 - C12))
        return TildeC

    def TildeCWrapper(self, tau, theta, tau_bar, phi, phi0, theta0, WN, WR, omega, f):
        """
        """
        T = omega.shape[0]
        g1n = self.gN(tau[0], theta[0], WN, f, self.NU[0], self.NU[1], self.N, T)
        g2n = self.gN(tau[1], theta[1], WN, f, self.NU[0], self.NU[1], self.N, T)
        g3n = self.gN(tau[2], theta[2], WN, f, self.NU[0], self.NU[1], self.N, T)
        TildeCN = self.TildeC(g1n, g2n, g3n)

        g1r = self.gR(tau_bar[0], phi[0], theta[0], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        g2r = self.gR(tau_bar[1], phi[1], theta[1], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        g3r = self.gR(tau_bar[2], phi[2], theta[2], phi0, theta0, WR, omega, f, self.NU[0], self.NU[1], self.N, T)
        TildeCR = self.TildeC(g1r, g2r, g3r)
        return TildeCN, TildeCR

    def ComputeTildeC(self, Phi, sU, rcs, hpars, verboseEst=False, bounds=False, run_fisher=False, run_detect=True, **kwargs):
        """
        """
        self.verboseEst = verboseEst
        self.bounds = bounds

        Phi, ChPars, _, prior, _ = self.MainSetup(Phi, sU, rcs, hpars, **kwargs)

        # Simulate signal model
        _, _, WN, WR, omega, f = self.main_signal_model(Phi, sU, rcs, prior)
        f = f[0, :]

        if self.verboseEst is True:
            if Phi.shape[0] > 0:
                print("Channel coefficients SB: \n", self.alphal)
                print("Channel coefficients DB: \n", self.alphal_bar)

        TildeCN, TildeCR \
            = self.TildeCWrapper(ChPars[:, 0]*1e-09, ChPars[:, 1:3], ChPars[:, 3]*1e-09, ChPars[:, 4:],
                                 prior["phi0"], prior["theta0"], WN, WR, omega, f)
        return TildeCN, TildeCR



if __name__ == "__main__":
    pass
