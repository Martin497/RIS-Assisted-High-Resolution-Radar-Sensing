# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 07:56:43 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implementation of orthogonal matching pursuit. (10/06/2024)
"""


import numpy as np
from numpy.linalg import norm, pinv
import matplotlib.pyplot as plt


def orthogonal_matching_pursuit(A, y, stopping_criteria=dict(), plotting=dict()):
    """
    Orthogonal mathcing pursuit algorithm for computing sparse input vector
    x_hat given measurement data y and a dictionary A.

    Parameters
    ----------
    A : ndarray, size=(N \times M)
        Dictionary.
    y : ndarray, size=N
        Measurement data.
    stopping_criteria : dict, optional
        The stopping criteria.
    plotting : dict, optional
        Options used for plotting.

    Returns
    -------
    x_hat[-1] : ndarray, size=M
        Sparse input vector.
    error : list
        List of errors during algorithm execution.
    t : int
        Number of iterations used.
    """
    eps1 = stopping_criteria["eps1"]
    sparsity = stopping_criteria["sparsity"]

    if str(A.dtype) == 'complex128':
        dtype = np.complex128
        y = y.astype(dtype)
    elif str(y.dtype) == 'complex128':
        dtype = np.complex128
        A = A.astype(dtype)
    else:
        dtype = np.float64
    m, n = np.shape(A)
    x_hat = []
    x_hat.append(np.zeros(n, dtype=dtype))
    r = []
    r.append(y)
    error = []
    error.append(norm(r[0]))
    Lambda = []
    t = 0
    while error[t] > eps1:
        receive_power = np.abs(np.einsum("mn,m->n", A.conj(), r[t], optimize="greedy"))
        if plotting["plot"] is True:
            receive_power_plot(plotting["delays"], plotting["az_angles"], plotting["el_angles"],
                               receive_power, title=f"OMP{t}")
        J = np.argmax(receive_power)
        if J in Lambda:
            break
        Lambda.append(J)
        x_hat_update = np.zeros(n, dtype=dtype)
        x_hat_update[Lambda] = np.dot(pinv(A[:, Lambda]), y)
        x_hat.append(x_hat_update)
        r.append(y - np.dot(A[:, Lambda], x_hat_update[Lambda]))
        t += 1
        error.append(norm(r[t]))
        if t == sparsity:
            print("OMP stopped by sparsity.")
            break
    if error[t] <= eps1:
        print("OMP stopped by small residual norm.")
    x_hat = x_hat[-1]
    return x_hat, Lambda


def receive_power_plot(delays, az_angles, el_angles, P, title="OMP"):
    """
    Plot the specified pseudo spectrum.
    """
    P = P.reshape((len(delays), len(az_angles), len(el_angles)))

    if len(delays) != 1:
        az_delay_P_dB = 20*np.log10(np.abs(np.mean(P, axis=2)))
        spec = plt.pcolormesh(az_angles, delays*1e09, az_delay_P_dB, cmap='viridis', shading='auto')
        cb = plt.colorbar(spec)
        cb.set_label(label='Amplitude [dB]')
        plt.xlabel('Azimuth [rad]')
        plt.ylabel('Delay [ns]')
        plt.title(title)
        plt.savefig(f"results/OMP_plots/{title}az_delay.png", dpi=500, bbox_inches="tight")
        plt.show()

        el_delay_P_dB = 20*np.log10(np.abs(np.mean(P, axis=1)))
        spec = plt.pcolormesh(el_angles, delays*1e09, el_delay_P_dB, cmap='viridis', shading='auto')
        cb = plt.colorbar(spec)
        cb.set_label(label='Amplitude [dB]')
        plt.xlabel('Elevation [rad]')
        plt.ylabel('Delay [ns]')
        plt.title(title)
        plt.savefig(f"results/OMP_plots/{title}el_delay.png", dpi=500, bbox_inches="tight")
        plt.show()

    el_az_P = np.abs(np.mean(P, axis=0))
    el_az_P_dB = 20*np.log10(el_az_P)
    spec = plt.pcolormesh(el_angles, az_angles, el_az_P_dB, cmap='viridis', shading='auto')
    cb = plt.colorbar(spec)
    cb.set_label(label='Amplitude [dB]')
    plt.xlabel('Elevation [rad]')
    plt.ylabel('Azimuth [rad]')
    plt.title(title)
    plt.savefig(f"results/OMP_plots/{title}el_az.png", dpi=500, bbox_inches="tight")
    plt.show()
