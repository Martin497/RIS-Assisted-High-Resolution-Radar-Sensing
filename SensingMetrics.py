# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:50:40 2023

@author: Martin Voigt Vejling
E-mail: mvv@es.aau.dk
"""


import numpy as np


def GOSPA(X_in, Y_in, p=2, c=2, alpha=2):
    """
    Computing the Generalized optimal sub-pattern assignment metric (GOSPA).
    This is a metric used for quantifying the distance between point patterns
    that can potentially have different cardinalities. For reference see

    "Generalized optimal sub-pattern assignment metric" (2017),
    By Abu Sajana Rahmathullah, Angel F. Garcia-Fernandez, Lennart Svensson.

    Parameters
    ----------
    X_in : ndarray, size=(n, d)
        Point pattern of n samples in d dimensional space.
    Y_in : ndarray, size=(m, d)
        Point pattern of m samples in d dimensional space.
    p : float
        Power exponent parameter. Higher values penalise outliers more heavily.
        The default is 2.
    c : float
        Maximum allowable localization error. Along with alpha this
        determines the error due to cardinality mismatch. A lower value penalizes
        mismatch in cardinality less. The default is 2.
    alpha : float
        Normalization of cardinality penalty. When alpha increases,
        the GOSPA metric decreases and vice versa. The default is 2.

    Returns
    -------
    GOSPA_distance : float
        The generalized optimal sub-pattern assignment metric (GOSPA)
        between point patterns X_in and Y_in.
    """
    # # Handle cases where one or both of the sets are empty
    # if len(X_in.shape) == 0 and len(Y_in.shape) == 0:
    #     return 0
    # elif len(X_in.shape) == 0:
    #     X_in = np.repeat(np.expand_dims(X_in, axis=1), Y_in.shape[1], axis=1)
    # elif len(Y_in.shape) == 0:
    #     Y_in = np.repeat(np.expand_dims(Y_in, axis=1), X_in.shape[1], axis=1)

    # Ensure that the cardinality of X is less than that of Y.
    if X_in.shape[0] <= Y_in.shape[0]:
        X, Y = X_in, Y_in
    else:
        X, Y = Y_in, X_in
    n, m = X.shape[0], Y.shape[0]

    if n > 0 and m > 0:
        dist = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        dist_c = np.minimum(dist, np.ones((n, m))*c)
        sum_term = np.sum(dist_c[np.arange(n), np.argmin(dist_c, axis=1)]**p)
        card_penalty = c**p/alpha*(m-n)
        GOSPA_distance = (sum_term + card_penalty)**(1/p)
    elif n == 0 and m == 0:
        GOSPA_distance = 0
    elif n == 0 and m > 0:
        GOSPA_distance = (c**p/alpha*m)**(1/p)
    return GOSPA_distance

def OSPA(X_in, Y_in, p=2, c=2):
    """
    Computing the optimal sub-pattern assignment metric (OSPA).
    This is a metric used for quantifying the distance between point patterns
    that can potentially have different cardinalities. For reference see

    "A Consistent Metric for Performance Evaluation of Multi-Object Filters" (2008),
    By Schuhmacher, Dominic and Vo, Ba-Tuong and Vo, Ba-Ngu.

    Parameters
    ----------
    X_in : ndarray, size=(n, d)
        Point pattern of n samples in d dimensional space.
    Y_in : ndarray, size=(m, d)
        Point pattern of m samples in d dimensional space.
    p : float
        Power exponent parameter. Higher values penalise outliers more heavily.
        The default is 2.
    c : float
        Maximum allowable localization error. This determines the error due to cardinality
        mismatch. A lower value penalizes mismatch in cardinality less. The default is 2.

    Returns
    -------
    OSPA_distance : float
        The optimal sub-pattern assignment metric (OSPA)
        between point patterns X_in and Y_in.
    """
    # Ensure that the cardinality of X is less than that of Y.
    if X_in.shape[0] <= Y_in.shape[0]:
        X, Y = X_in, Y_in
    else:
        X, Y = Y_in, X_in
    n, m = X.shape[0], Y.shape[0]

    if n > 0 and m > 0:
        dist = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        dist_c = np.minimum(dist, np.ones((n, m))*c)
        sum_term = np.sum(dist_c[np.arange(n), np.argmin(dist_c, axis=1)]**p)
        card_penalty = c**p*(m-n)
        OSPA_distance = (1/m*(sum_term + card_penalty))**(1/p)
    elif n == 0 and m == 0:
        OSPA_distance = 0
    elif n == 0 and m > 0:
        OSPA_distance = c
    return OSPA_distance

