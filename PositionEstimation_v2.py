# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:34:00 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Base functionality to do weighted non-linear least squares. (13/02/2024)
"""


import numpy as np

from scipy.optimize import minimize
from system import channel


class PosEst(channel):
    """
    """
    def __init__(self, config_file, **kwargs):
        """
        """
        super(PosEst, self).__init__(config_file, False, **kwargs)

    def etaParsPos(self, p, sU, sR):
        """
        Compute and organize channel parameters in an array given a position p.
        """
        pU, oU = sU[:3], sU[3:]
        pR, oR = sR[:3], sR[3:]

        R_UE = self.rot(*oU)
        p_SU = np.einsum("ij,i->j", R_UE, p - pU)
        theta_az = np.arctan2(p_SU[1], p_SU[0])
        theta_el = np.arccos(p_SU[2]/np.linalg.norm(p_SU))

        R_RIS = self.rot(*oR)
        p_SR = np.einsum("ij,i->j", R_RIS, p - pR)
        phi_az = np.arctan2(p_SR[1], p_SR[0])
        phi_el = np.arccos(p_SR[2]/np.linalg.norm(p_SR))

        d_SU = np.linalg.norm(p - pU)
        tau = 2*d_SU/self.c*1e09

        d_UR = np.linalg.norm(pU - pR)
        d_SR = np.linalg.norm(p - pR)
        d_SU = np.linalg.norm(p - pU)
        tau_bar = (d_UR + d_SR + d_SU)/self.c*1e09

        eta = np.array([tau, theta_az, theta_el, tau_bar, phi_az, phi_el])
        return eta

    def funPos(self, p, sU, sR, eta_hat, Prec):
        """
        Evaluate the object function: compute the l2-norm between the channel
        parameters eta of the current proposal for a position p and the estimated
        channel parameters eta_hat.
        """
        eta = self.etaParsPos(p, sU, sR)
        fun = 1/2 * np.dot(eta - eta_hat, np.dot(Prec, eta - eta_hat))
        return fun

    def jacPos(self, p, sU, sR, eta_hat, Prec):
        """
        Evaluate the Jacobian of funPos: compute the derivatives with respect
        to the channel parameters eta.
        """
        pU, oU = sU[:3], sU[3:]
        pR, oR = sR[:3], sR[3:]

        diff_SU = p - pU
        diff_RS = pR - p
        diff_SR = p - pR

        R_UE = self.rot(*oU)
        p_SU = np.einsum("ij,i->j", R_UE, p - pU)
        R_RIS = self.rot(*oR)
        p_SR = np.einsum("ij,i->j", R_RIS, p - pR)

        eta = self.etaParsPos(p, sU, sR)
        diff = eta - eta_hat
        derThetaAz = (R_UE[1, :] * p_SU[0] - R_UE[0, :] * p_SU[1]) / (p_SU[0]**2 + p_SU[1]**2)
        derThetaEl = - (R_UE[2, :] * np.linalg.norm(p_SU)**2 - p_SU[2] * diff_SU) / (np.linalg.norm(p_SU)**3 * np.sqrt(1 - (p_SU[2]/np.linalg.norm(p_SU))**2))
        derPhiAz = (R_RIS[1, :] * p_SR[0] - R_RIS[0, :] * p_SR[1]) / (p_SR[0]**2 + p_SR[1]**2)
        derPhiEl = - (R_RIS[2, :] * np.linalg.norm(p_SR)**2 - p_SR[2] * diff_SR) / (np.linalg.norm(p_SR)**3 * np.sqrt(1 - (p_SR[2]/np.linalg.norm(p_SR))**2))
        derTau = 2*diff_SU /(self.c * np.linalg.norm(diff_SU))*1e09
        derTauBar = diff_RS/(self.c * np.linalg.norm(diff_RS)) + diff_SU/(self.c * np.linalg.norm(diff_SU))*1e09
        der = np.array([derTau, derThetaAz, derThetaEl, derTauBar, derPhiAz, derPhiEl])

        jac = np.array([1/2 * (np.dot(d, np.dot(Prec, diff)) + np.dot(diff, np.dot(Prec, d))) for d in der.T])
        return jac

    def PositionEstimation(self, sU, sR, eta_hat, p0, Prec=None):
        """
        Estimate the position using both non-RIS and RIS estimated channel
        parameters to minimizing the l2-norm in the channel parameter space
        using scipy.optimize.minimize.
        """
        if Prec is None:
            # Sigma = np.eye(eta_hat.shape[0])
            Prec = np.diag((1, 1, 1, 1e-10, 1, 1))
            # Prec = np.diag((1, 1, 1, 1, 1, 1))
        res = minimize(self.funPos, p0, args=(sU, sR, eta_hat, Prec), jac=self.jacPos)
        Phi_est = res["x"]
        return Phi_est

    def RISetaParsPos(self, p, sU, sR):
        """
        Compute and organize channel parameters in an array given a position p.
        """
        pU, _ = sU[:3], sU[3:]
        pR, oR = sR[:3], sR[3:]

        R_RIS = self.rot(*oR)
        p_SR = np.einsum("ij,i->j", R_RIS, p - pR)
        phi_az = np.arctan2(p_SR[1], p_SR[0])
        phi_el = np.arccos(p_SR[2]/np.linalg.norm(p_SR))

        d_UR = np.linalg.norm(pU - pR)
        d_SR = np.linalg.norm(p - pR)
        d_SU = np.linalg.norm(p - pU)
        tau_bar = (d_UR + d_SR + d_SU)/self.c*1e09

        eta = np.array([tau_bar, phi_az, phi_el])
        return eta

    def RISfunPos(self, p, sU, sR, eta_hat, Prec):
        """
        Evaluate the object function: compute the l2-norm between the channel
        parameters eta of the current proposal for a position p and the estimated
        channel parameters eta_hat.
        """
        eta = self.RISetaParsPos(p, sU, sR)
        fun = 1/2 * np.dot(eta - eta_hat, np.dot(Prec, eta - eta_hat))
        return fun

    def RISjacPos(self, p, sU, sR, eta_hat, Prec):
        """
        Evaluate the Jacobian of funPos: compute the derivatives with respect
        to the channel parameters eta.
        """
        pU, _ = sU[:3], sU[3:]
        pR, oR = sR[:3], sR[3:]

        diff_SU = p - pU
        diff_RS = pR - p
        diff_SR = p - pR

        R_RIS = self.rot(*oR)
        p_SR = np.einsum("ij,i->j", R_RIS, p - pR)

        eta = self.RISetaParsPos(p, sU, sR)
        diff = eta - eta_hat
        derPhiAz = (R_RIS[1, :] * p_SR[0] - R_RIS[0, :] * p_SR[1]) / (p_SR[0]**2 + p_SR[1]**2)
        derPhiEl = - (R_RIS[2, :] * np.linalg.norm(p_SR)**2 - p_SR[2] * diff_SR) / (np.linalg.norm(p_SR)**3 * np.sqrt(1 - (p_SR[2]/np.linalg.norm(p_SR))**2))
        derTauBar = diff_RS/(self.c * np.linalg.norm(diff_RS)) + diff_SU/(self.c * np.linalg.norm(diff_SU))*1e09
        der = np.array([derTauBar, derPhiAz, derPhiEl])

        jac = np.array([1/2 * (np.dot(d, np.dot(Prec, diff)) + np.dot(diff, np.dot(Prec, d))) for d in der.T])
        return jac

    def RISPositionEstimation(self, sU, sR, eta_hat, p0, Prec=None, tau_bar_adjust=None):
        """
        Estimate the position using RIS estimated channel
        parameters to minimizing the l2-norm in the channel parameter space
        using scipy.optimize.minimize.
        """
        if Prec is None:
            Prec = np.diag((1e-02, 1, 1))
            # Prec = np.diag((1, 1, 1))
        if tau_bar_adjust is not None:
            Prec[0, 0] = tau_bar_adjust # 1e04*Prec[0, 0] # the EFIM of the DB delay is underestimated
        res = minimize(self.RISfunPos, p0, args=(sU, sR, eta_hat, Prec), jac=self.RISjacPos)
        Phi_est = res["x"]
        return Phi_est
