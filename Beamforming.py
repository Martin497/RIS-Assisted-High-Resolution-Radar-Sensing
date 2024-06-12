# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 08:50:31 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Making the module with functions from the methods in
           MainEstimation.py ChannelEstimation class.
           This module contains functionality to do spatial smoothing for
           signals in 4D space (x, y, f, t), compute the Bartlett, Capon,
           and MUSIC pseudo-spectra on a grid of points, and find peaks
           in the pseudo-spectrum using a maximum filter. (06/12/2023)
    v1.1 - Flexible pseudo-spectrum plotting. Added functionality to find peaks.
           Optimizing numpy einsum. (21/12/2023)
    v1.2 - Added functionality to find peaks. (28/03/2024)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


def forward_smoothing(Y, L_sx, L_sy, L_sf, L_st):
    """
    Forward smoothing.
    """
    T, N, NUx, NUy = Y.shape
    P_x = NUx-L_sx + 1 #Number of subarrays in x-direction
    P_y = NUy-L_sy + 1 #Number of subarrays in y-direction
    P_f = N-L_sf + 1 #Number of sbuarrays in f-direction
    P_t = T-L_st + 1 #Number of sbuarrays in t-direction
    R_hat = np.zeros((L_sx*L_sy*L_sf*L_st, L_sx*L_sy*L_sf*L_st), dtype=np.complex128)
    for px in range(P_x):
        for py in range(P_y):
            for pf in range(P_f):
                for pt in range(P_t):
                    Y_sub = Y[pt:pt+L_st, pf:pf+L_sf, px:px+L_sx, py:py+L_sy]
                    Y_sub_flat = Y_sub.flatten()
                    R_hat += np.outer(Y_sub_flat, Y_sub_flat.conj())
    R_hat = 1/(P_x*P_y*P_f*P_t) * R_hat
    return R_hat

def smoothing(Rf):
    """
    Implementing forward-backward smoothing.
    """
    dim = Rf.shape[0]
    J = np.eye(dim)[::-1]
    R_hat = 1/2 * (Rf + np.matmul(J, np.matmul(Rf.conj(), J)))
    return R_hat

def BartlettSpectrum(H, R):
    """
    Compute the Bartlett pseudo spectrum.

    Input
    -----
        H : ndarray, size=(K_delay, K_az, K_el, Lsx*Lsy*Lsf)
            The response vector in each search direction.
        R : ndarray, size=(Lsx*Lsy*Lsf, Lsx*Lsy*Lsf)
            The estimated precision matrix.
    """
    denominator = np.einsum("...i,...i->...", H.conj(), H, optimize="greedy")**2
    numerator = np.einsum("...j,...j->...", H.conj(), np.einsum("ji,...i->...j", R, H, optimize="greedy"), optimize="greedy")
    P_bartlett = numerator/denominator
    return P_bartlett

def CaponSpectrum(H, R):
    """
    Compute the Capon pseudo spectrum.

    Input
    -----
        H : ndarray, size=(K_delay, K_az, K_el, Lsx*Lsy*Lsf)
            The response vector in each search direction.
        R : ndarray, size=(Lsx*Lsy*Lsf, Lsx*Lsy*Lsf)
            The estimated precision matrix.
    """
    Rinv = np.linalg.inv(R)
    prod = np.einsum("...j,...j->...", H.conj(), np.einsum("ji,...i->...j", Rinv, H, optimize="greedy"), optimize="greedy")
    P_capon = 1/prod
    return P_capon

def MUSICSpectrum(H, R, M):
    """
    Compute the MUSIC pseudo spectrum.

    Input
    -----
        H : ndarray, size=(K_delay, K_az, K_el, Lsx*Lsy*Lsf)
            The response vector in each search direction.
        R : ndarray, size=(Lsx*Lsy*Lsf, Lsx*Lsy*Lsf)
            The estimated precision matrix.
        M : int
            Dimension of signal subspace.
    """
    v, U = np.linalg.eigh(R) # Compute eigendecomposition
    Un = U[:, :-M] # Define the noise subspace

    # Compute the pseudo spectrum
    prod1 = np.einsum("ij,...i->...j", Un.conj(), H, optimize="greedy")
    P_music = 1/np.einsum("...j,...j->...", prod1.conj(), prod1, optimize="greedy")
    return P_music

def PseudoSpectrum_plot(delays, az_angles, el_angles, P, savename, title="MUSIC"):
    """
    Plot the specified pseudo spectrum.
    """
    P = P/np.max(P)

    if len(delays) != 1:
        az_delay_P_dB = 20*np.log10(np.abs(np.sum(P, axis=2)))
        spec = plt.pcolormesh(az_angles, delays*1e09, az_delay_P_dB, cmap='viridis', shading='auto')
        cb = plt.colorbar(spec)
        cb.set_label(label='Amplitude [dB]')
        plt.xlabel('Azimuth [rad]')
        plt.ylabel('Delay [ns]')
        plt.title(title)
        plt.savefig(savename+"_az_delay.png", dpi=500, bbox_inches="tight")
        plt.show()

        el_delay_P_dB = 20*np.log10(np.abs(np.sum(P, axis=1)))
        spec = plt.pcolormesh(el_angles, delays*1e09, el_delay_P_dB, cmap='viridis', shading='auto')
        cb = plt.colorbar(spec)
        cb.set_label(label='Amplitude [dB]')
        plt.xlabel('Elevation [rad]')
        plt.ylabel('Delay [ns]')
        plt.title(title)
        plt.savefig(savename+"_el_delay.png", dpi=500, bbox_inches="tight")
        plt.show()

    el_az_P = np.abs(np.sum(P, axis=0))
    el_az_P_dB = 20*np.log10(el_az_P)
    spec = plt.pcolormesh(el_angles, az_angles, el_az_P_dB, cmap='viridis', shading='auto')
    cb = plt.colorbar(spec)
    cb.set_label(label='Amplitude [dB]')
    plt.xlabel('Elevation [rad]')
    plt.ylabel('Azimuth [rad]')
    plt.title(title)
    plt.savefig(savename+"_el_az.png", dpi=500, bbox_inches="tight")
    plt.show()

def find_peaks(P, ChPars, stds=2, kernel=(3, 3, 3), number_of_peaks=None, return_local_maxima=False):
    """
    Find peaks in pseudo spectrum P. Peaks are defined to be above a
    threhold th = mean(P) * std(P)*stds and to be greater than neighbors.

    Input
    -----
        P : ndarray, size=(...)
            The pseudo spectrum.
        ChPars : ndarray, size=(..., 3)
            The parameters for each point in the pseudo spectrum grid.
        stds : float
            The peak detection threshold parameter.
        kernel : tuple, len=...
            The kernel used for the maximum filter. Must be of same dimension
            as P. Minimum values for the tuple entries is 3 which defines
            a maximum filter on first order neighbors only.

    Return
    ------
        est : ndarray, size=(?, 3)
            The estimated channel parameters. The number of channel
            parameter sets depends on the number of detected peaks.
    """
    max_filter = maximum_filter(P, size=kernel, mode='constant', cval=1e10)
    mask_loc = P >= max_filter
    if return_local_maxima is True:
        return P[mask_loc], ChPars[mask_loc], np.mean(P), np.std(P)
    if number_of_peaks is None:
        try:
            th = np.max(P[mask_loc])*stds
            mask_th = P > th
            mask = np.logical_and(mask_th, mask_loc)
        except ValueError:
            mask = mask_loc
    else:
        if np.sum(mask_loc) > number_of_peaks:
            mask = np.copy(mask_loc)
            lower_indices = np.argpartition(P[mask_loc], -number_of_peaks)[:-number_of_peaks]
            mask_temp = mask[mask_loc]
            mask_temp[lower_indices] = False
            mask[mask_loc] = mask_temp
        else:
            mask = np.copy(mask_loc)
    est = ChPars[mask]
    return est