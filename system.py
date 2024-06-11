# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 07:47:06 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Initial implementation is an adaptation from the RIS_Aided_Sensing
           channel model followed by a refactoring to fit the purpose.
           Includes functionality to:
               - Compute non-RIS and RIS channel parameters
               - Compute array response and delay response.
               - Construct precoders, RIS phase profiles, and combiners.
               - Simulate from the two-step protocol signal model.
           Also includes beampattern plotting functionality. (17/10/23)
    v1.1 - Implement **kwargs initialization option. Correctly pass around the
           UE state including the orientation. Include some tensor options
           for fast parallelization. (21/12/2023)
    v1.2 - Include option to simulate radar cross sections. (23/02/2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import toml

import sys
import os
if os.path.abspath("..")+"/Modules" not in sys.path:
    sys.path.append(os.path.abspath("..")+"/Modules")
from utilities import make_3D_grid


class channel(object):
    """
    Channel model for the RIS-enabled High-Resolution Sensing
    Two-Step Protocol.

    Methods
    -------
        __init__ : Initialize settings.

    Attributes
    ----------
        verbose : bool
            If true show plots.


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
        include_interference : bool
            If True, include interference terms.
    """
    def __init__(self, config_file, verbose, **kwargs):
        """
        Initialize the class:
            - Load configuration file.
            - Specify frequency parameters, antennas positions, and
              power parameters.

        Inputs
        ------
            config_file : str
                Path to the .toml configuration file.
            verbose : bool
                If True, do plots, else if False, do no plots.

        Keyword arguments
        ------------------
            Settings for the channel. These are used unless a config file path
            is provided, then they can be overwritten.
        """
        self.verbose  = verbose

        ### Load data from keyword arguments ###
        self.fc = kwargs.get("fc", 3e09)
        self.p_tot_tx_dbm = kwargs.get("p_tot_tx_dbm", 130)
        self.p_noise_hz_dbm = kwargs.get("p_noise_hz_dbm", -100)
        self.sR = np.array(kwargs.get("sR", [22, 8.5, 12, 0, 0, 0]))
        self.nR = np.array(kwargs.get("nR", [0, 0, 1]))
        self.sU = np.array(kwargs.get("sU", [0, 0, 0, 0, 0, 0]))
        self.NU = kwargs.get("NU", [4, 4])
        self.NR = kwargs.get("NR", [25, 25])
        self.delta_f = kwargs.get("delta_f", 5e05)
        self.N = kwargs.get("N", 15)
        self.c = kwargs.get("c", 299792458)
        self.T1 = kwargs.get("T1", 20)
        self.T2 = kwargs.get("T2", 20)
        # self.window = kwargs.get("window", [0, 50, 0, 20, 0, 10])
        self.fading = kwargs.get("fading", "rayleigh")
        self.rcs_model = kwargs.get("rcs_model", "swerling1")

        ### Load and interpret .toml file ###
        if config_file is not None:
            self.toml_in = toml.load(config_file)
            toml_settings = self.toml_in["settings"]
            for key, value in toml_settings.items():
                setattr(self, key, value)

        self.pR, self.oR = np.array(self.sR[:3]), np.array(self.sR[3:])
        self.pU, self.oU = np.array(self.sU[:3]), np.array(self.sU[3:])

        #### Frequency parameters ###
        self.W = self.delta_f*self.N # Bandwidth
        if self.N % 2 == 0:
            self.n = np.arange(-self.N//2, self.N//2) # n=-N/2,...,N/2-1
        else:
            self.n = np.arange(-(self.N-1)//2, (self.N-1)//2+1) # n=-(N-1)/2,...,(N-1)/2

        ### Wavelength and antenna array spacing ###
        self.lambda_ = self.c/self.fc # Wavelength
        self.dR = self.lambda_/4 # Grating lobes at the RIS are avoided with antenna spacing lambda/4.
        self.dU = self.lambda_/2

        ### Antenna positions ###
        self.NU_prod = np.prod(self.NU)
        self.NR_prod = np.prod(self.NR)

        self.AntPos_UE = self.AntenPos(self.NU[0], self.NU[1], "UE")
        self.AntPos_RIS = self.AntenPos(self.NR[0], self.NR[1], "RIS")

        ### Power ###
        self.p_tot_tx = 10**(self.p_tot_tx_dbm/10 - 3) # total transmit power in W
        self.p_tx = self.p_tot_tx/(self.N*(self.T1+self.T2)*self.NU_prod) # transmit power per sub-carrier per antenna per time in W
        self.p_tx_dbm = 10*np.log10(self.p_tx) + 30 # in dBm

        self.p_noise_hz = 10**(self.p_noise_hz_dbm/10 - 3) # noise power per Hz in W
        self.p_noise_sc = self.p_noise_hz*self.delta_f # noise power per sub-carrier in W
        self.p_noise_sc_dbm = 10*np.log10(self.p_noise_sc) + 30 # in dBm
        self.p_noise = self.p_noise_sc*self.N # total noise power in W
        self.p_noise_dbm = 10*np.log10(self.p_noise) + 30 # in dBm

        # print(np.sqrt(self.p_noise/2 * 25*75*2*2))

        # std = np.sqrt(self.p_noise/2)/np.sqrt(2)
        # real_eps = np.random.normal(0, std, size=(25, 75, 2, 2))
        # imag_eps = np.random.normal(0, std, size=(25, 75, 2, 2))
        # eps = real_eps + 1j*imag_eps
        # print(np.linalg.norm(eps))

        if self.verbose is True:
            print(f"Transmit power: {self.p_tx_dbm:.2f} dBm")
            print(f"Noise power: {self.p_noise_dbm:.2f} dBm")
            print(f"Transmit SNR: {10*np.log10(self.p_tx/self.p_noise):.2f} dB")

    def __str__(self):
        """
        """

    # def step1_signal_model(self, Phi, sU, include_interference, theta0=None, T=None):
    #     """
    #     Simulate from the signal model for the first step of the protocol.

    #     Inputs
    #     ------
    #         Phi : ndarray, size=(L, 3)
    #         sU : ndarray, size=(6,)
    #         include_interference : bool
    #         theta0 : ndarray, size=(2,)
    #             Direction for precoding focus. If None, use random precoding
    #             sampled from a DFT matrix.
    #         T : int
    #             The number of OFDM symbols.
    #     """
    #     if T is None:
    #         T = self.T1//2

    #     omega = self.RIS_random_codebook(T=T) # size=(T1/2, NR)
    #     f = self.construct_precoder(theta0=theta0, T=T) # size=(T1/2, NU)

    #     if self.verbose is True:
    #         print(self.ChParams(Phi, sU))
    #         self.plot_precoder(f)
    #         self.plot_RIS(omega, self.RIS_angles(Phi, sU[:3])[0])

    #     H_RIS, H_nonRIS = self.ChMat(Phi, sU, omega, include_interference=True) # size=(T1/2, N, NU, NU)
    #     y_nonRIS = self.simulate_received_signal(H_nonRIS, f, np.sqrt(self.p_noise/2)) # size=(T1/2, N, NU)
    #     y_RIS = self.simulate_received_signal(H_RIS, f, np.sqrt(self.p_noise/2)) # size=(T1/2, N, NU)
    #     return y_nonRIS, y_RIS

    # def step2_signal_model_oracle(self, Phi, sU, include_interference):
    #     """
    #     Simulate from the signal model for the second step of the protocol
    #     assuming known RIS and SP cluster position.

    #     Inputs
    #     ------
    #         Phi : ndarray, size=(L, 3)
    #         sU : ndarray, size=(6,)
    #         include_interference : bool
    #     """
    #     # Oracle angles
    #     theta0, thetal = self.nonRIS_angles(Phi, sU)
    #     phi0, phil = self.RIS_angles(Phi, sU[:3])

    #     # Compute RIS phase profiles
    #     phil_min = np.min(phil, axis=0)
    #     phil_max = np.max(phil, axis=0)
    #     phi_bounds = (phil_min[0]-0.2, phil_max[0]+0.2, phil_min[1]-0.2, phil_max[1]+0.2)
    #     omega = self.RIS_directional(phi0, phi_bounds, method="pencil") # size=(T2, NR)

    #     # Simulate signal model
    #     f = self.construct_precoder(theta0, thetal[0], T=self.T2) # size=(T2, NU)
    #     H_RIS, H_nonRIS = self.ChMat(Phi, sU, omega, include_interference=False) # size=(T2, N, NU, NU)
    #     H = H_RIS + H_nonRIS
    #     W = self.construct_combiner(theta0) # size=(NU,)

    #     if self.verbose is True:
    #         self.plot_precoder(f)
    #         self.plot_RIS(omega, phi0)

    #     y = self.simulate_received_signal(H, f, np.sqrt(self.p_noise)) # size=(T2, N, NU)
    #     y = np.einsum("ij,tni->tnj", np.conjugate(W), y) # size=(T2, N, NU-1)
    #     return y, omega, phi_bounds

    # def step2_signal_model(self, Phi, sU, include_interference, ChParsEst_nonRIS, PosEst_nonRIS, ChParsEst_RIS):
    #     """
    #     Simulate from the signal model for the second step of the protocol
    #     using estimates for the RIS and SP cluster positions.

    #     OBS! The method assumes detection of only 1 SP in the first step.

    #     Inputs
    #     ------
    #         Phi : ndarray, size=(L, 3)
    #         sU : ndarray, size=(6,)
    #         include_interference : bool
    #         ChParsEst_nonRIS : ndarray, size=(1, 3)
    #             The channel parameters estimated in the first step of the SPs
    #             organized as (delay, azimuth, elevation).
    #         PosEst_nonRIS : ndarray, size=(1, 3)
    #             The estimated position of SPs in euclidean coordinates.
    #         ChParsEst_RIS : ndarray, size=(1, 3)
    #             The channel parameters for the RIS estimated in the first step
    #             organized as (delay, azimuth, elevation).
    #     """
    #     # Compute RIS phase profiles
    #     ResRegion = self.sensor_resolution_function(PosEst_nonRIS, sU)
    #     phi_bounds = self.RIS_angle_resolution_region(sU[:3], ResRegion)
    #     omega = self.RIS_directional(ChParsEst_RIS[0, 1:], phi_bounds) # size=(T2, NR)

    #     # Simulate signal model
    #     f = self.construct_precoder(ChParsEst_RIS[0, 1:], ChParsEst_nonRIS[0, 1:], T=self.T2) # size=(T2, NU)
    #     H_RIS, H_nonRIS = self.ChMat(Phi, sU, omega, include_interference=False) # size=(T2, N, NU, NU)
    #     H = H_RIS + H_nonRIS
    #     W = self.construct_combiner(ChParsEst_RIS[0, 1:]) # size=(NU,)

    #     if self.verbose is True:
    #         self.plot_precoder(f)
    #         self.plot_RIS(omega, self.RIS_angles(Phi, sU[:3])[0])

    #     y = self.simulate_received_signal(H, f, np.sqrt(self.p_noise)) # size=(T2, N, NU)
    #     y = np.einsum("ij,tni->tnj", np.conjugate(W), y) # size=(T2, N, NU-1)
    #     return y, omega, phi_bounds

    def prior_step_signal_model(self, Phi, sU, rcs=None):
        """
        Inputs:
        -------
            Phi : ndarray, size=(L, 3)
            sU : ndarray, size=(6,)
            rcs : ndarray, size=(L,)

        Output:
        -------
            yN : ndarray, size=(T1, N, NU)
        """
        self.pU, self.oU = np.array(self.sU[:3]), np.array(self.sU[3:])
        self.AntPos_UE = self.AntenPos(self.NU[0], self.NU[1], "UE")

        f = 1/np.sqrt(self.NU_prod) * np.ones((self.T1//2, self.NU_prod))
        # f = self.construct_precoder(np.array([0.7, 0.8]), T=self.T2//2)
        # self.construct_precoder(T=self.T1//2)
        WN = np.eye(self.NU_prod)

        if Phi.shape[0] > 0:
            H_USU, _ = self.nonRIS_channel_matrix(Phi, sU, rcs, T=self.T1//2)
            temp = self.simulate_received_signal(H_USU, f, np.sqrt(self.p_noise/2))
            yN = np.einsum("ij,tni->tnj", np.conjugate(WN), temp) # size=(T2, N, NU)
        elif Phi.shape[0] == 0:
            self.alphal = np.array([])

            std = np.sqrt(self.p_noise/2)/np.sqrt(2)
            real_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            imag_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            yN = real_eps + 1j*imag_eps
        if self.verbose is True:
            self.plot_precoder(f)
        return yN, WN, f

    def main_signal_model(self, Phi, sU, rcs, prior):
        """
        The main signal model for RIS-assisted high-resolution sensing.
        In this signal model, we assume the start-up phase has been conducted
        prior giving the dict "prior".
        Then, time-orthogonal RIS phase profiles are used where
        the design of the RIS phases makes a beam sweep over the prior defined
        angles. The precoder is towards the SP cluster from the prior and with
        a null towards the RIS. We make the combiner as an identity (for now).
        Using the RIS orthogonality we can construct a RIS signal where we only
        have contribution from paths UE-SP-RIS-UE and a non-RIS signal with
        the UE-SP-UE paths.

        Inputs:
        -------
            Phi : ndarray, size=(L, 3)
            sU : ndarray, size=(6,)
            rcs : ndarray, size=(L,)
            prior : dict
                thetal : ndarray, size=(2,)
                theta0 : ndarray, size=(2,)
                phi0 : ndarray, size=(2,)
                phi_bounds : ndarray, size=(4,)

        Output:
        -------
            yR : ndarray, size=(T2//2, N, NU)
            yN : ndarray, size=(T2//2, N, NU)
        """
        self.pU, self.oU = np.array(self.sU[:3]), np.array(self.sU[3:])
        self.AntPos_UE = self.AntenPos(self.NU[0], self.NU[1], "UE")

        # f = self.construct_precoder(prior["thetal"], prior["theta0"], T=self.T2//2)
        f = self.construct_precoder(prior["thetal"], T=self.T2//2)
        # f = np.ones((self.T2//2, self.NU_prod))
        # f = self.construct_precoder(T=self.T2//2)
        WN = np.eye(self.NU_prod)
        WR = np.eye(self.NU_prod) # self.construct_precoder(prior["theta0"], T=1)
        omega = self.RIS_directional(prior["phi0"], prior["phi_bounds"], method="uncertainty_region", T=self.T2//2)

        if Phi.shape[0] > 0:
            H_USU, _ = self.nonRIS_channel_matrix(Phi, sU, rcs, T=self.T2//2)
            temp = self.simulate_received_signal(H_USU, f, np.sqrt(self.p_noise/2))
            yN = np.einsum("ij,tni->tnj", np.conjugate(WN), temp) # size=(T2, N, NU)
    
            _, _, H_USRU, _ = self.RIS_channel_matrix(Phi, sU, omega, rcs)
            temp = self.simulate_received_signal(H_USRU, f, np.sqrt(self.p_noise/2))
            yR = np.einsum("ij,tni->tnj", np.conjugate(WR), temp) # size=(T2, N, NU)
        elif Phi.shape[0] == 0:
            self.alphal, self.alphal_bar = np.array([]), np.array([])

            # Simulate noise
            std = np.sqrt(self.p_noise/2)/np.sqrt(2)
            real_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            imag_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            yN = real_eps + 1j*imag_eps

            real_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            imag_eps = np.random.normal(0, std, size=(self.T2//2, self.N, self.NU_prod))
            yR = real_eps + 1j*imag_eps
        if self.verbose is True:
            self.plot_precoder(f)
            self.plot_RIS(omega, phi0)
        return yN, yR, WN, WR, omega, f

    def sensor_resolution_function(self, p, sU, res_scale=5):
        """
        Constructing the resolution region.

        Inputs
        ------
            p : ndarray, size=(3,)
            sU : ndarray, size=(6,)
            res_scale : float
                The factor we improve over the theoretical resolution.
        """
        ### Input preparation ###
        pU, oU = sU[:3], sU[3:]
        Naz, Nel = self.NU

        ### Find channel parameters ###
        difference = p - pU
        distance = np.linalg.norm(difference)
        delay = 2*distance/self.c

        rotation_matrix = self.rot(*oU)
        p_SU = np.einsum("ij,i->j", rotation_matrix, difference) # size=(3,)

        theta_az = np.arctan2(p_SU[1], p_SU[0])
        theta_el = np.arccos(p_SU[2]/np.linalg.norm(p_SU))

        ### Construct resolution region ###
        delay_resolution = 1/(self.W*res_scale)
        az_resolution = 2/((Naz-1)*np.abs(np.sin(theta_az))*res_scale)
        el_resolution = 2/((Nel-1)*np.abs(np.sin(theta_el))*res_scale)
        # az_resolution = 2/((Naz-1)*np.abs(np.cos(theta_az))*res_scale)
        # el_resolution = 2/((Nel-1)*np.abs(np.cos(theta_el))*res_scale)
        delay_interval = np.array([delay-delay_resolution/2, delay+delay_resolution/2])
        az_interval = np.array([theta_az-az_resolution/2, theta_az+az_resolution/2])
        el_interval = np.array([theta_el-el_resolution/2, theta_el+el_resolution/2])

        ResRegion = np.zeros((2, 2, 2, 3))
        for idx1, delay in enumerate(delay_interval):
            for idx2, az in enumerate(az_interval):
                for idx3, el in enumerate(el_interval):
                    ResRegion[idx1, idx2, idx3, :] = self.ChParsToEuc(delay*1e09, az, el, sU)
        ResRegion = ResRegion.reshape((-1, 3))
        return ResRegion

    def ChParsToEuc(self, delay, az, el, sU):
        """
        Converting channel parameters at the UE to Euclidean positions.
        """
        pU, oU = sU[:3], sU[3:]
        dist = delay/2*1e-09*self.c
        rotation_matrix = self.rot(*oU) # size=(3, 3)
        pos = pU + dist*np.einsum("ij,j->i", rotation_matrix,
                                  np.array([np.cos(az)*np.sin(el), np.sin(az)*np.sin(el), np.cos(el)]))
        return pos

    def RIS_angle_resolution_region(self, pU, ResRegion):
        """
        Computing the bounds for the RIS angle in the resolution region
        based on the Euclidean position of the boundaries of the resolution region.
        """
        R = self.rot(*self.oR) # size=(3, 3)

        # Find RIS-target angle
        p_SR = np.einsum("ij,ni->nj", R, ResRegion - self.pR[None, :]) # size=(8, 3)
        phil_az = np.arctan2(p_SR[:, 1], p_SR[:, 0]) # size=(8,)
        phil_el = np.arccos(p_SR[:, 2]/np.linalg.norm(p_SR, axis=-1)) # size=(8,)

        # Create angular uncertainty region
        phi_az_min = np.min(phil_az)
        phi_az_max = np.max(phil_az)
        phi_el_min = np.min(phil_el)
        phi_el_max = np.max(phil_el)
        return phi_az_min, phi_az_max, phi_el_min, phi_el_max

    def RIS_delays(self, p, pU):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        d_UR = np.linalg.norm(pU - self.pR)
        d_SR = np.linalg.norm(p - self.pR, axis=-1)
        d_SU = np.linalg.norm(p - pU, axis=-1)
        tau0 = np.array([2*d_UR/self.c])
        taul_bar = (d_UR + d_SR + d_SU)/self.c
        taul_2bar = (2*d_UR + 2*d_SR)/self.c
        return tau0, taul_bar, taul_2bar

    def nonRIS_delays(self, p, pU):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        d_SU = np.linalg.norm(p - pU, axis=-1)
        d_SS = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

        taul = 2*d_SU/self.c # size=(L, )

        taul_tilde = (2*d_SU[:, None] + d_SS)/self.c
        taul_tilde = taul_tilde[~np.eye(taul_tilde.shape[0], dtype=bool)].reshape(taul_tilde.shape[0], -1) # size=(L, L-1)
        return taul, taul_tilde

    def small_scale_fading(self, sz, sigma=1):
        """
        """
        if self.fading == "rayleigh":
            nu = np.random.normal(0, sigma/np.sqrt(2), size=sz) + 1j*np.random.normal(0, sigma/np.sqrt(2), size=sz)
        elif self.fading == "random_phase":
            nu = np.exp(-1j*2*np.pi*np.random.uniform(0, 1, size=sz))
        elif self.fading == "rayleigh_mean":
            nu = sigma * np.sqrt(np.pi/2)
        else:
            nu = sigma
        return nu

    def radar_cross_section(self, rcs):
        if self.rcs_model == "rayleigh":
            sigma_rcs = np.random.rayleigh(scale=rcs * np.sqrt(2/np.pi))
        elif self.rcs_model == "rayleigh_mean":
            sigma_rcs = rcs
        elif self.rcs_model == "swerling1":
            sigma_rcs = np.random.chisquare(2)
        return sigma_rcs

    def RIS_channel_gains(self, p, pU, rcs=None, q0=0.285):
        """
        """
        L, d = p.shape
        if rcs is None:
            rcs = np.sqrt(50)*np.ones(L)

        tau0, taul_bar, taul_2bar = self.RIS_delays(p, pU)

        d_UR = np.linalg.norm(pU - self.pR)
        d_SR = np.linalg.norm(p - self.pR, axis=-1)
        d_SU = np.linalg.norm(p - pU, axis=-1)

        g_UR = np.abs(np.dot(pU - self.pR, self.nR)/d_UR)
        g_SR = np.abs(np.dot(p - self.pR, self.nR)/d_SR)

        alpha0_amp = np.sqrt(self.lambda_**2 * g_UR**(4*q0) / ((4*np.pi)**3 * d_UR**4))

        alphal_bar_amp_mean = np.sqrt(self.lambda_**4 * g_UR**(2*q0) * g_SR**(2*q0) * rcs**2 \
                                      / ((4*np.pi)**4 * d_UR**2 * d_SR**2 * d_SU**2))
        self.alphal_bar_amp_mean = alphal_bar_amp_mean
        self.Palphal_bar = np.sqrt(self.p_tx * self.lambda_**4 * g_UR**(2*q0) * g_SR**(2*q0) * rcs**2 / ((4*np.pi)**4 * d_UR**2 * d_SR**2 * d_SU**2))
        self.PL_bar = np.sqrt(self.lambda_**4 * g_UR**(2*q0) * g_SR**(2*q0) / ((4*np.pi)**4 * d_UR**2 * d_SR**2 * d_SU**2))

        sigma_rcs = np.zeros(L)
        for l in range(L):
            sigma_rcs[l] = self.radar_cross_section(rcs[l])
        self.RIS_sigma_rcs = sigma_rcs

        alphal_bar_amp = np.sqrt(self.lambda_**4 * g_UR**(2*q0) * g_SR**(2*q0) * sigma_rcs**2 \
                                 / ((4*np.pi)**4 * d_UR**2 * d_SR**2 * d_SU**2))
        self.alphal_bar_amp = alphal_bar_amp

        alphal_2bar_amp = np.sqrt(self.lambda_**6 * g_UR**(2*q0) * g_SR**(4*q0) * sigma_rcs**2 \
                                  / ((4*np.pi)**5 * d_UR**4 * d_SR**4))
        return alpha0_amp, alphal_bar_amp, alphal_2bar_amp

    def RIS_channel_coefficients(self, p, pU, rcs=None, q0=0.285):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        L, d = p.shape
        tau0, taul_bar, taul_2bar = self.RIS_delays(p, pU)
        alpha0_amp, alphal_bar_amp, alphal_2bar_amp = self.RIS_channel_gains(p, pU, rcs, q0)

        nu = self.small_scale_fading((1), 1)
        alpha0 = alpha0_amp * np.exp(-1j*(2*np.pi*self.fc*tau0)) * nu

        nu = self.small_scale_fading((L), 1)
        alphal_bar = alphal_bar_amp * np.exp(-1j*(2*np.pi*self.fc*taul_bar)) * nu
        self.alphal_bar = alphal_bar

        nu = self.small_scale_fading((L), 1)
        alphal_2bar = alphal_2bar_amp * np.exp(-1j*(2*np.pi*self.fc*taul_2bar)) * nu
        return alpha0, alphal_bar, alphal_2bar

    def nonRIS_channel_gains(self, p, pU, rcs=None):
        """
        """
        L, d = p.shape

        if rcs is None:
            rcs = 50*np.ones(L)

        taul, taul_tilde = self.nonRIS_delays(p, pU)

        d_SU = np.linalg.norm(p - pU, axis=-1)
        d_SS = np.linalg.norm(p[None, :, :] - p[:, None, :], axis=-1)
        d_SS = d_SS[~np.eye(d_SS.shape[0], dtype=bool)].reshape(d_SS.shape[0], -1) # size=(L, L-1)

        alphal_amp_mean = np.sqrt(self.lambda_**2 * rcs**2 / ((4*np.pi)**3 * d_SU**4)) # size=(L,)
        self.alphal_amp_mean = alphal_amp_mean
        self.Palphal = np.sqrt(self.p_tx * self.lambda_**2 * rcs**2 / ((4*np.pi)**3 * d_SU**4))
        self.PL = np.sqrt(self.lambda_**2 / ((4*np.pi)**3 * d_SU**4))

        sigma_rcs = np.zeros(L)
        for l in range(L):
            sigma_rcs[l] = self.radar_cross_section(rcs[l])
        self.nonRIS_sigma_rcs = sigma_rcs

        alphal_amp = np.sqrt(self.lambda_**2 * sigma_rcs**2 / ((4*np.pi)**3 * d_SU**4)) # size=(L,)
        self.alphal_amp = alphal_amp

        rcs_SS = sigma_rcs[:, None]**2 * sigma_rcs[None, :]**2
        rcs_SS = rcs_SS[~np.eye(rcs_SS.shape[0], dtype=bool)].reshape(rcs_SS.shape[0], -1) # size=(L, L-1)

        alphal_tilde_amp = np.sqrt(self.lambda_**4 * rcs_SS / ((4*np.pi)**4 * d_SU[:, None]**4 * d_SS)) # size=(L, L-1)
        return alphal_amp, alphal_tilde_amp

    def nonRIS_channel_coefficients(self, p, pU, rcs=None):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        L, d = p.shape
        taul, taul_tilde = self.nonRIS_delays(p, pU)

        alphal_amp, alphal_tilde_amp = self.nonRIS_channel_gains(p, pU, rcs)

        nu = self.small_scale_fading((L), 1)
        alphal = alphal_amp * np.exp(-1j*(2*np.pi*self.fc*taul)) * nu
        self.alphal = alphal

        nu = self.small_scale_fading((L, L-1), 1)
        alphal_tilde = alphal_tilde_amp * np.exp(-1j*(2*np.pi*self.fc*taul_tilde)) * nu
        return alphal, alphal_tilde

    def RIS_angles(self, p, pU):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        R_RIS = self.rot(*self.oR)

        # Compute difference vectors in local coordinates
        p_UR = np.einsum("ij,i->j", R_RIS, pU - self.pR)
        p_SR = np.einsum("ij,li->lj", R_RIS, p - self.pR[None, :])

        # Compute the azimuth and elevation angles
        phi0_az = np.arctan2(p_UR[1], p_UR[0])
        phi0_el = np.arccos(p_UR[2]/np.linalg.norm(p_UR))
        phi0 = np.array([phi0_az, phi0_el])

        phil_az = np.arctan2(p_SR[:, 1], p_SR[:, 0])
        phil_el = np.arccos(p_SR[:, 2]/np.linalg.norm(p_SR, axis=-1))
        phil = np.stack((phil_az, phil_el), axis=-1)
        return phi0, phil

    def nonRIS_angles(self, p, sU):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        pU, oU = sU[:3], sU[3:]
        R_UE = self.rot(*oU)

        # Compute difference vectors in local coordinates
        p_RU = np.einsum("ij,i->j", R_UE, self.pR - pU)
        p_SU = np.einsum("ij,li->lj", R_UE, p - pU[None, :])

        # Compute the azimuth and elevation angles
        theta0_az = np.arctan2(p_RU[1], p_RU[0])
        theta0_el = np.arccos(p_RU[2]/np.linalg.norm(p_RU))
        theta0 = np.array([theta0_az, theta0_el])

        thetal_az = np.arctan2(p_SU[:, 1], p_SU[:, 0])
        thetal_el = np.arccos(p_SU[:, 2]/np.linalg.norm(p_SU, axis=-1))
        thetal = np.stack((thetal_az, thetal_el), axis=-1)
        return theta0, thetal

    def ChParams(self, p, sU, rcs=None, q0=0.285):
        """
        Inputs
        ------
            p : ndarray, size=(L, 3)
                The SP positions.
            pU : ndarray, size=(3,)
                The UE position.
        """
        pU, oU = sU[:3], sU[3:]
        tau0, taul_bar, taul_2bar = self.RIS_delays(p, pU)
        taul, taul_tilde = self.nonRIS_delays(p, pU)
        alpha0, alphal_bar, alphal_2bar = self.RIS_channel_coefficients(p, pU, rcs, q0)
        alphal, alphal_tilde = self.nonRIS_channel_coefficients(p, pU, rcs)
        theta0, thetal = self.nonRIS_angles(p, sU)
        phi0, phil = self.RIS_angles(p, pU)
        pars = {"tau0": tau0, "taul_bar": taul_bar, "taul_2bar": taul_2bar,
                "taul": taul, "taul_tilde": taul_tilde,
                "alpha0": alpha0, "alphal_bar": alphal_bar, "alphal_2bar": alphal_2bar,
                "alphal": alphal, "alphal_tilde": alphal_tilde,
                "theta0": theta0, "thetal": thetal,
                "phi0": phi0, "phil": phil}
        return pars

    def ChParamsVec(self, p, sU):
        """
        """
        pU, oU = sU[:3], sU[3:]
        L = p.shape[0]
        _, taul_bar, _ = self.RIS_delays(p, pU)
        taul, _ = self.nonRIS_delays(p, pU)
        _, thetal = self.nonRIS_angles(p, sU)
        _, phil = self.RIS_angles(p, pU)
        ChParsVec = np.zeros((L, 6))
        for l in range(L):
            ChParsVec[l] = np.array([taul[l], thetal[l, 0], thetal[l, 1], taul_bar[l], phil[l, 0], phil[l, 1]])
        return ChParsVec

    def rot(self, yaw, pitch, roll):
        """
        """
        R = self.rotz(yaw) @ self.roty(pitch) @ self.rotx(roll)
        return R

    def rotz(self, angle):
        """
        """
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        return R

    def roty(self, angle):
        """
        """
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        return R

    def rotx(self, angle):
        """
        """
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])
        return R

    # def AntenPos(self, N_az, N_el, type_):
    #     """
    #     Generating uniform plarnar antenna (UPA).
    #     UPA lies on x-z plane, not x-y plane.
    #     Antenna is heading to y.
    #     azimuth from +x axis (counter-clock); elevation from +z axis to down.
    #     """
    #     if type_ == "RIS":
    #         d = self.dR
    #         R = self.rot(*self.oR)
    #     elif type_ == "UE":
    #         d = self.dU
    #         R = self.rot(*oU)

    #     pos_local = np.zeros((N_az*N_el, 3))
    #     for i in range(N_az):
    #         for j in range(N_el):
    #             pos_local[i*N_el+j, :] = np.array([(N_az+1)/2 - (i+1), (N_el+1)/2 - (j+1), 0]) * d
    #     pos_global = np.einsum("ij,Nj->Ni", R, pos_local)

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(pos_local[:, 0], pos_local[:, 1], pos_local[:, 2], label="local")
    #     ax.scatter(pos_global[:, 0], pos_global[:, 1], pos_global[:, 2], label="global")
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     plt.legend()
    #     plt.show()
    #     return pos_global

    def AntenPos(self, N_az, N_el, type_):
        """
        Generating uniform plarnar antenna (UPA).
        UPA lies on x-y plane. Antenna is heading to z.
        Azimuth from +x axis (counter-clock); Elevation from +z axis to down.
        """
        if type_ == "RIS":
            d = self.dR
            R = self.rot(*self.oR)
        elif type_ == "UE":
            d = self.dU
            R = self.rot(*self.oU)

        pos_local = np.zeros((N_az*N_el, 3))
        for i in range(N_az):
            for j in range(N_el):
                pos_local[i*N_el+j, :] = np.array([i, j, 0]) * d
        pos_global = np.einsum("ij,Nj->Ni", R, pos_local)

        if self.verbose is True:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pos_local[:, 0], pos_local[:, 1], pos_local[:, 2], label="local")
            ax.scatter(pos_global[:, 0], pos_global[:, 1], pos_global[:, 2], label="global")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.legend()
            plt.show()
        return pos_global

    def WaveVec(self, az, el):
        """
        Wavenumber vector.
        """
        waveVec = 2*np.pi/self.lambda_ * np.array([np.cos(az)*np.sin(el), np.sin(az)*np.sin(el), np.cos(el)])
        return waveVec

    def ArrayVec(self, AntPos, WaveVec):
        """
        Uniform planar array response vector from antenna positions in global coordinates.
        """
        a = np.exp(1j * np.dot(AntPos, WaveVec))
        return a

    def ArrayVec_(self, AntPos, az, el):
        """
        Wrapper function...
        """
        WaveVec = self.WaveVec(az, el)
        a = self.ArrayVec(AntPos, WaveVec)
        return a

    def WaveVecTensor(self, az, el):
        """
        Wavenumber vector.
        """
        waveVec = 2*np.pi/self.lambda_ * np.stack((np.cos(az)[:, None]*np.sin(el)[None, :], np.sin(az)[:, None]*np.sin(el)[None, :], np.ones(len(az))[:, None]*np.cos(el)[None, :]), axis=-1)
        return waveVec

    def ArrayVecTensor(self, AntPos, WaveVec):
        """
        Uniform planar array response vector from antenna positions in global coordinates.
        """
        a = np.exp(1j * np.einsum("pi,...i->...p", AntPos, WaveVec, optimize="greedy"))
        return a

    def ArrayVecTensor_(self, AntPos, az, el):
        """
        Wrapper function...
        """
        WaveVec = self.WaveVecTensor(az, el)
        a = self.ArrayVecTensor(AntPos, WaveVec)
        return a

    def DelayVec(self, tau):
        """
        Compute delay steering vector given a delay tau of size=(L,).
        """
        sh = tau.shape
        dn_tau = np.exp(-1j*2*np.pi*self.n[:, None]*self.delta_f*tau.flatten()[None, :]).reshape((self.N, *sh))
        return dn_tau

    def nonRIS_channel_matrix(self, Phi, sU, rcs=None, T=None):
        """
        Inputs
        ------
            Phi : ndarray, size=(L, 3)
                The SP positions.
            sU : ndarray, size=(3,)
                The UE state
            rcs : ndarray, size=(L,)
                Radar cross section.
            T : int
                The number of channel uses.
        """
        if T is None:
            T = self.T1//2

        L = Phi.shape[0]
        pU = sU[:3]

        taul, taul_tilde = self.nonRIS_delays(Phi, pU)
        alphal, alphal_tilde = self.nonRIS_channel_coefficients(Phi, pU, rcs)
        theta0, thetal = self.nonRIS_angles(Phi, sU)

        # Compute array response vectors
        aU_thetal = np.zeros((L, self.NU_prod), dtype=np.complex128)
        for l, p in enumerate(Phi):
            waveVec = self.WaveVec(thetal[l, 0], thetal[l, 1])
            aU_thetal[l, :] = self.ArrayVec(self.AntPos_UE, waveVec) # size=(N_U,)
        AU_lk = np.einsum("li,kj->lkij", aU_thetal, aU_thetal) # size=(L, L, N_U, N_U)
        AU_ll = AU_lk[np.eye(AU_lk.shape[0], dtype=bool)].reshape(AU_lk.shape[0], *AU_lk.shape[2:]) # size=(L, N_U, N_U)
        AU_lk = AU_lk[~np.eye(AU_lk.shape[0], dtype=bool)].reshape(AU_lk.shape[0], AU_lk.shape[1]-1, *AU_lk.shape[2:]) # size=(L, L-1, N_U, N_U)

        # Compute delay response vectors
        dn_taul = self.DelayVec(taul) # size=(N, L)
        dn_taul_tilde = self.DelayVec(taul_tilde) # size=(N, L, L-1)

        # Compute channel matrix
        H = np.sum(alphal[None, None, :, None, None] * np.ones(T)[:, None, None, None, None] \
            * AU_ll[None, None, :, :, :] * dn_taul[None, :, :, None, None], axis=2) # size=(T, N, N_U, N_U)
        H_tilde = np.sum(alphal_tilde[None, None, :, :, None, None] * np.ones(T)[:, None, None, None, None, None] \
            * AU_lk[None, None, :, :, :, :] * dn_taul_tilde[None, :, :, :, None, None], axis=(2, 3)) # size=(T, N, N_U, N_U)
        return H, H_tilde

    def RIS_channel_matrix(self, Phi, sU, omega, rcs=None):
        """
        Inputs
        ------
            Phi : ndarray, size=(L, 3)
                The SP positions.
            sU : ndarray, size=(3,)
                The UE state
            omega : ndarray, size=(T, NR)
                The RIS phase profiles.
            include_UE_RIS_UE : bool
                If True, include the UE-RIS-UE path, else if False
                include only double bounce paths.
        """
        L = Phi.shape[0]
        pU = sU[:3]

        # Compute channel parameters
        tau0, taul_bar, taul_2bar = self.RIS_delays(Phi, pU)
        alpha0, alphal_bar, alphal_2bar = self.RIS_channel_coefficients(Phi, pU, rcs)
        theta0, thetal = self.nonRIS_angles(Phi, sU)
        phi0, phil = self.RIS_angles(Phi, pU)

        # Compute array response vectors
        aU_theta0 = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]) # size=(NU,)
        aR_phi0 = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1]) # size=(NR,)

        aU_thetal = np.zeros((L, self.NU_prod), dtype=np.complex128)
        for l, p in enumerate(Phi):
            waveVec = self.WaveVec(thetal[l, 0], thetal[l, 1])
            aU_thetal[l, :] = self.ArrayVec(self.AntPos_UE, waveVec) # size=(NU,)

        aR_phil = np.zeros((L, self.NR_prod), dtype=np.complex128)
        for l, p in enumerate(Phi):
            waveVec = self.WaveVec(phil[l, 0], phil[l, 1])
            aR_phil[l, :] = self.ArrayVec(self.AntPos_RIS, waveVec) # size=(NR,)

        # Compute RIS response
        aR_phi0_prod = omega*aR_phi0[None, :]
        nu_00 = np.einsum("i,ti->t", aR_phi0, aR_phi0_prod) # size=(T,)
        nu_l0 = np.einsum("li,ti->tl", aR_phil, aR_phi0_prod) # size=(T, L)

        # Compute delay steering vectors
        dn_tau0 = self.DelayVec(tau0)[:, 0] # size=(N,)
        dn_taul_bar = self.DelayVec(taul_bar) # size=(N, L)
        dn_taul_2bar = self.DelayVec(taul_2bar) # size=(N, L)

        # Compute array response
        AU_00 = np.einsum("i,j->ij", aU_theta0, aU_theta0) # size=(N_U, N_U)
        AU_l0 = np.einsum("li,j->lij", aU_thetal, aU_theta0) # size=(L, N_U, N_U)
        AU_0l = np.einsum("i,lj->lij", aU_theta0, aU_thetal) # size=(L, N_U, N_U)

        # Compute channel matrix components
        H0 = alpha0 * nu_00[:, None, None, None] * AU_00[None, None, :, :] * dn_tau0[None, :, None, None] # size=(T, N, N_U, N_U)

        H1 = np.sum(alphal_bar[None, None, :, None, None] * nu_l0[:, None, :, None, None] \
            * AU_l0[None, None, :, :, :] * dn_taul_bar[None, :, :, None, None], axis=2) # size=(T, N, N_U, N_U)

        H2 = np.sum(alphal_bar[None, None, :, None, None] * nu_l0[:, None, :, None, None] \
            * AU_0l[None, None, :, :, :] * dn_taul_bar[None, :, :, None, None], axis=2) # size=(T, N, N_U, N_U)

        H3 = np.sum(alphal_2bar[None, None, :, None, None] * (nu_l0**2)[:, None, :, None, None] \
            * AU_00[None, None, None, :, :] * dn_taul_2bar[None, :, :, None, None], axis=2) # size=(T, N, N_U, N_U)
        return H0, H1, H2, H3

    def ChMat(self, Phi, sU, omega, include_interference=True):
        """
        Wrapper function...
        """
        T = omega.shape[0]
        H_URU, H_URSU, H_USRU, H_URSRU = self.RIS_channel_matrix(Phi, sU, omega)
        H_USU, H_USSU = self.nonRIS_channel_matrix(Phi, sU, T)
        if include_interference is True:
            H_RIS = H_URU + H_URSU + H_USRU + H_URSRU
            H_nonRIS = H_USU + H_USSU
        elif include_interference is False:
            H_RIS = H_URSU
            H_nonRIS = H_USU
        return H_RIS, H_nonRIS

    def construct_precoder(self, theta0=None, theta1=None, T=None):
        """
        Construct the precoder. Strategy depending on input...

        Inputs
        ------
            theta0 : ndarray, size=(2,)
                The [az, el] angle for the directional precoding.
            theta1 : ndarray, size=(2,)
                Null towards [az, el] for the precoding.
            T : int
                The number of channel uses.
        """
        if T is None:
            T = self.T1//2

        if theta0 is None and theta1 is None:
            f = 1/np.sqrt(self.NU_prod) * np.exp(1j*2*np.pi*np.random.rand(T, self.NU_prod)) # size=(T, N_U)
        elif theta1 is None:
            aU_theta0 = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]) # size=(N_U,)
            f = 1/np.sqrt(self.NU_prod) * np.conjugate(aU_theta0)[None, :] * np.ones(T)[:, None] # size=(T, N_U)
        else:
            aU_theta0 = self.ArrayVec_(self.AntPos_UE, theta0[0], theta0[1]) # size=(N_U,)
            aU_theta1 = self.ArrayVec_(self.AntPos_UE, theta1[0], theta1[1]) # size=(N_U,)
            f_temp = 1/np.sqrt(self.NU_prod) * aU_theta0[None, :] * np.ones(T)[:, None] # size=(T, N_U)
            proj = 1/self.NU_prod * np.einsum("i,ti->t", np.conjugate(aU_theta1), f_temp)[:, None] * aU_theta1[None, :] # size=(T, N_U)
            f = np.conjugate(f_temp - proj)/np.linalg.norm(f_temp - proj, axis=1)[:, None]
        return f

    def construct_combiner(self, theta):
        """
        Construct the combiner matrix. The columns form an orthonormal
        basis for the space of vectors with a null towards theta=[az, el].
        """
        aU_theta = self.ArrayVec_(self.AntPos_UE, theta[0], theta[1]) # size=(N_U,)
        DFT_idx = np.arange(self.NU_prod)[None, :]*np.arange(self.NU_prod)[:, None]
        DFTt = np.exp(1j*2*np.pi*DFT_idx/self.NU_prod) / np.sqrt(self.NU_prod)
        W = (DFTt * aU_theta[:, None])[:, 1:]
        return W

    def RIS_random_codebook(self, T=None):
        """
        Construct a sequence of random RIS phase profiles with phases in the
        interval [-\pi, \pi]. The number of channel uses is T.
        """
        if T is None:
            T = self.T1//2

        random_phases = np.random.uniform(low=-np.pi, high=np.pi, size=(T, self.NR_prod))
        omega_tilde = np.exp(1j*random_phases)
        return omega_tilde

    def RIS_grid(self, phi_bounds, T=None):
        """
        """
        if T is None:
            T = self.T2

        phi_az_min, phi_az_max, phi_el_min, phi_el_max = phi_bounds

        # Split the (azimuth, elevation) domain into T2 rectangular sub-regions
        N = np.ceil(np.sqrt(T))
        mod_ = -1
        N -= 1
        while mod_ != 0:
            N += 1
            mod_ = T % N
        az_fac = int(N)
        el_fac = int(T/N)
        assert az_fac*el_fac == T, "The number of channel uses must have a decomposition into two integers!"

        if self.verbose is True:
            print(f"Number of phi discretization points: {az_fac} , {el_fac}")

        phi_az_min_arr = np.linspace(phi_az_min, phi_az_max, az_fac, endpoint=False)
        # phi_az_min_arr += (phi_az_min_arr[1] - phi_az_min_arr[0])/2
        phi_el_min_arr = np.linspace(phi_el_min, phi_el_max, el_fac, endpoint=False)
        # phi_el_min_arr += (phi_el_min_arr[1] - phi_el_min_arr[0])/2
        C1, C2 = np.meshgrid(phi_az_min_arr, phi_el_min_arr, indexing="ij")
        phi_min_grid_flat = np.stack((C1.flatten(), C2.flatten()), axis = 1) # size=(T2, 2)
        return phi_az_min_arr, phi_el_min_arr, phi_min_grid_flat, az_fac, el_fac

    def RIS_directional(self, phi0, phi_bounds, method="uncertainty_region", T=None):
        """
        Directional RIS phase profiles based on prior information.

        Input
        -----
            phi0 : ndarray, size=(2,)
                The AoA at the RIS from the UE.
            phi_bounds : tuple, len=4
                The bounds of the resolution region in the phi parameter
                    az_min, az_max, el_min, el_max.
            method : str
                The method. Options are "uncertainty_region" and "pencil".
        """
        # phi_bounds[0] -= 0.08
        # phi_bounds[2] -= 0.08
        # phi_bounds = [0, np.pi, 0, np.pi/2]
        if self.verbose is True:
            print("RIS angles bounds: ", phi_bounds)
        if T is None:
            T = self.T2

        aR_phi0 = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1]) # size=(NR,)

        Naz, Nel = self.NR
        n = np.arange(1, self.NR_prod+1)
        n_el = n - Naz*(np.ceil(n/Naz).astype(np.int16) - 1)
        n_az = np.ceil(n/Nel).astype(np.int16)

        ### phi0 ###
        # phi0_az_min, phi0_az_max, phi0_el_min, phi0_el_max = phi0[0]-0.2, phi0[0]+0.2, phi0[1]-0.2, phi0[1]+0.2
        # phi0_bounds = (phi0_az_min, phi0_az_max, phi0_el_min, phi0_el_max)
        # phi0_az_min_arr, phi0_el_min_arr, phi0_min_grid_flat, az0_fac, el0_fac = self.RIS_grid(phi0_bounds)

        # zeta0_az = (phi0_az_max - phi0_az_min)/az0_fac
        # zeta0_el = (phi0_el_max - phi0_el_min)/el0_fac

        ### phil ###
        phi_az_min, phi_az_max, phi_el_min, phi_el_max = phi_bounds
        phi_az_min_arr, phi_el_min_arr, phi_min_grid_flat, az_fac, el_fac = self.RIS_grid(phi_bounds, T)

        zeta_az = (phi_az_max - phi_az_min)/az_fac
        zeta_el = (phi_el_max - phi_el_min)/el_fac

        if method == "uncertainty_region":
            # Define target angles: phi0 #
            # phi0_az = phi0_min_grid_flat[:, 0, None] + zeta0_az*(n_az[None, :]-1)/(Naz-1) # size=(T2, NR)
            # phi0_el = phi0_min_grid_flat[:, 1, None] + zeta0_el*(n_el[None, :]-1)/(Nel-1) # size=(T2, NR)
    
            # beta0 = np.exp(1j*2*np.pi*np.cos(phi0_az)*np.sin(phi0_el)*self.dR/self.lambda_)**(n_az[None, :]-1) # size=(T2, NR)
            # gamma0 = np.exp(1j*2*np.pi*np.sin(phi0_az)*np.sin(phi0_el)*self.dR/self.lambda_)**(n_el[None, :]-1) # size=(T2, NR)
            # a0 = beta0*gamma0 # size=(NR,)
    
            # Define target angles: phil #
            phi_az = phi_min_grid_flat[:, 0, None] + zeta_az*(n_az[None, :]-1)/(Naz-1) # size=(T2, NR)
            phi_el = phi_min_grid_flat[:, 1, None] + zeta_el*(n_el[None, :]-1)/(Nel-1) # size=(T2, NR)
    
            beta = np.exp(1j*2*np.pi*np.cos(phi_az)*np.sin(phi_el)*self.dR/self.lambda_)**(n_az[None, :]-1) # size=(T2, NR)
            gamma = np.exp(1j*2*np.pi*np.sin(phi_az)*np.sin(phi_el)*self.dR/self.lambda_)**(n_el[None, :]-1) # size=(T2, NR)
            a_dir = beta*gamma # size=(NR,)
    
            omega = np.conjugate(aR_phi0[None, :] * a_dir) # size=(T2, N_R)
            # omega = np.conjugate(a0 * a_dir) # size=(T2, N_R)
        elif method == "pencil":
            # Define target angles
            phi_az = phi_min_grid_flat[:, 0] + zeta_az # size=(T2,)
            phi_el = phi_min_grid_flat[:, 1] + zeta_el # size=(T2,)

            a_dir = np.zeros((T, self.NR_prod), dtype=np.complex128) # size=(T2, NR)
            for t in range(T):
                a_dir[t, :] = self.ArrayVec_(self.AntPos_RIS, phi_az[t], phi_el[t])

            # beta = np.exp(1j*2*np.pi*np.cos(phi_az[:, None])*np.sin(phi_el[:, None])*self.dR/self.lambda_)**(n_az[None, :]-1) # size=(NR,)
            # gamma = np.exp(1j*2*np.pi*np.sin(phi_az[:, None])*np.sin(phi_el[:, None])*self.dR/self.lambda_)**(n_el[None, :]-1) # size=(NR,)
            # a_dir2 = beta*gamma # size=(T2, NR)
            # print(np.allclose(a_dir, a_dir2))

            omega = np.conjugate(aR_phi0[None, :] * a_dir) # size=(T2, NR)
        return omega

    def simulate_received_signal(self, H, f, sigma):
        """
        Simulate from the signal model.

        Inputs
        ------
            H : ndarray, size=(T, N, NU, NU)
                The channel matrix.
            f : ndarray, size=(T, NU)
                The precoder.
            sigma : float
                The standard deviation of the complex noise.
        """
        T, N, _, _ = H.shape

        # Simulate noise
        std = sigma/np.sqrt(2)
        real_eps = np.random.normal(0, std, size=(T, N, self.NU_prod))
        imag_eps = np.random.normal(0, std, size=(T, N, self.NU_prod))
        eps = real_eps + 1j*imag_eps

        # Compute received signal
        y = np.sqrt(self.p_tx)*np.einsum("tnij,tj->tni", H, f) + eps
        return y

    def plot_precoder(self, f):
        """
        """
        azimuth = np.linspace(-np.pi, np.pi, 360//4)
        elevation = np.linspace(0, np.pi, 180//4)

        a_angle = np.zeros((len(azimuth), len(elevation), self.NU_prod), dtype=np.complex128)
        for idx1, az in enumerate(azimuth):
            for idx2, el in enumerate(elevation):
                a_angle[idx1, idx2, :] = self.ArrayVec_(self.AntPos_UE, az, el) # size=(N_U,)

        amp = 10*np.log10(np.sum(np.abs(np.einsum("aei,ti->tea", a_angle, f))**2, axis=0))
        self.beampattern_map(azimuth, elevation, amp, "Amplitude [dB]", "Precoder")

    def beampattern_map(self, azimuth, elevation, amp, label, title):
        """
        """
        spec = plt.pcolormesh(azimuth, elevation, amp, cmap="cool", shading="auto")
        cb = plt.colorbar(spec)
        cb.set_label(label=label)
        plt.xlabel("Azimuth [rad]")
        plt.ylabel("Elevation [rad]")
        plt.title(title)
        plt.show()

    def plot_RIS(self, omega, phi0):
        """
        """
        T = omega.shape[0]

        azimuth = np.linspace(-np.pi, np.pi, 360//2)
        elevation = np.linspace(0, np.pi, 180//2)

        nu = np.zeros((T, len(azimuth), len(elevation)), dtype=np.complex128)
        for idx1, az in enumerate(azimuth):
            for idx2, el in enumerate(elevation):
                a0 = self.ArrayVec_(self.AntPos_RIS, phi0[0], phi0[1]) # size=(NR,)
                adir = self.ArrayVec_(self.AntPos_RIS, az, el) # size=(NR,)
                nu[:, idx1, idx2] = np.einsum("ti,i->t", omega, (a0 * adir)) # size=(T,)

        amp_sum = 10*np.log10(np.sum(np.abs(nu)**2, axis=0))
        self.beampattern_map(azimuth, elevation, amp_sum.T, "Amplitude [dB]", "RIS")

        # amp = 10*np.log10(np.abs(nu)**2)
        # for t in range(T):
        #     self.beampattern_map(azimuth, elevation, amp[t, :, :].T, "Amplitude [dB]", f"RIS{t+1}")

    def plot_response(self, y, ground_truth):
        """
        """
        freq = (self.fc + self.n*self.delta_f)*1e-09
        plt.plot(freq, np.real(np.sum(y[:, :, 0], axis=0)), color="tab:blue", label="Real part")
        plt.plot(freq, np.imag(np.sum(y[:, :, 0], axis=0)), color="tab:red", label="Imaginary part")
        plt.ylabel("Frequency response [W]")
        plt.xlabel("Frequency [GHz]")
        plt.legend()
        plt.show()
    
        time = np.arange(self.N)/self.W*1e09
        for gt in ground_truth:
            plt.axvline(gt, color="red")
        plt.plot(time[:50], np.abs(np.fft.ifft(np.sum(y[:, :, 0], axis=0)))[:50], color="tab:blue")
        plt.ylabel("Impulse response [W]")
        plt.xlabel("Time [ns]")
        plt.show()


if __name__ == '__main__': # run demo
    chn = channel("system_config.toml", True)
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
    sR = np.array(chn.sR)
    lambda_ = chn.lambda_
    NR = chn.NR

    Phi_taus = np.array(toml_positions["Phi_taus"])
    Phi_rs = Phi_taus * 1e-09 * toml_settings["c"]
    Phi_azs = np.array(toml_positions["Phi_azs"])
    Phi_els = np.array(toml_positions["Phi_els"])
    Phi = Phi_rs[:, None] * np.stack((np.cos(Phi_azs)*np.sin(Phi_els), np.sin(Phi_azs)*np.sin(Phi_els), np.cos(Phi_els)), axis=-1)
    print("Targets: \n", Phi)
    triu_idx = np.triu_indices(Phi.shape[0], k=1)
    dists = np.linalg.norm(Phi[None, :, :] - Phi[:, None, :], axis=2)
    print("Distances between targets: \n", dists[triu_idx])
    dist_RIS = np.linalg.norm(sR[None, :3] - Phi, axis=1)
    print("Distance to RIS: \n", dist_RIS)
    Fraunhofer_dist = 2*(NR[0]*lambda_/4)**2/lambda_
    print("Fraunhofer distance: \n", Fraunhofer_dist)
    rcs = np.array(toml_positions["Phi_rcs"])

    # =============================================================================
    # Make prior
    # =============================================================================
    pU, oU = sU[:3], sU[3:]
    center = np.mean(Phi, axis=0)
    res_scale = 60
    ResRegion0 = chn.sensor_resolution_function(center, sU, 2*res_scale)
    ResRegion = chn.sensor_resolution_function(center, sU, res_scale)
    phi_bounds = chn.RIS_angle_resolution_region(pU, ResRegion0)
    phi0, _ = chn.RIS_angles(np.expand_dims(center, axis=0), sU[:3])
    theta0, thetal = chn.nonRIS_angles(np.expand_dims(center, axis=0), sU)
    _, theta_ResRegion = chn.nonRIS_angles(ResRegion, sU)
    tau_ResRegion, _ = chn.nonRIS_delays(ResRegion, pU)
    _, tau_bar_ResRegion, _ = chn.RIS_delays(ResRegion, pU)
    theta_bounds = (np.min(theta_ResRegion, axis=0)[0], np.max(theta_ResRegion, axis=0)[0],
                    np.min(theta_ResRegion, axis=0)[1], np.max(theta_ResRegion, axis=0)[1])
    tau_bounds = (np.min(tau_ResRegion), np.max(tau_ResRegion))
    tau_bar_bounds = (np.min(tau_bar_ResRegion), np.max(tau_bar_ResRegion))

    prior = {"thetal": thetal[0], "phi0": phi0, "theta0": theta0,
             "tau_bounds": tau_bounds, "theta_bounds": theta_bounds,
             "tau_bar_bounds": tau_bar_bounds, "phi_bounds": phi_bounds}

    # =============================================================================
    # Run signal model
    # =============================================================================
    yN, yR, WN, WR, omega, f = chn.main_signal_model(Phi, sU, rcs, prior)



