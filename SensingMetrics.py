# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:50:40 2023

@author: Martin Voigt Vejling
E-mail: mvv@es.aau.dk

Track changes:
    Version 1.0: Implementation of GOSPA. (19/01/2023)
            1.1: Implementation of OSPA (07/06/2023)
            1.2: Implementation of OSPA_extent. (22/08/2023)
"""


import numpy as np
from scipy.linalg import sqrtm


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
    # Ensure that the cardinality of X is less than that of Y.
    if X_in.shape[0] <= Y_in.shape[0]:
        X, Y = X_in, Y_in
    else:
        X, Y = Y_in, X_in
    n, m = X.shape[0], Y.shape[0]

    dist = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    dist_c = np.minimum(dist, np.ones((n, m))*c)
    sum_term = np.sum(dist_c[np.arange(n), np.argmin(dist_c, axis=1)]**p)
    card_penalty = c**p/alpha*(m-n)
    GOSPA_distance = (sum_term + card_penalty)**(1/p)
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


# def OSPA_extent(X_in, Y_in, p=2, c=2):
#     """
#     Computing the optimal sub-pattern assignment metric (OSPA).
#     This is a metric used for quantifying the distance between point patterns
#     that can potentially have different cardinalities. For reference see

#     "A Consistent Metric for Performance Evaluation of Multi-Object Filters" (2008),
#     By Schuhmacher, Dominic and Vo, Ba-Tuong and Vo, Ba-Ngu.

#     Parameters
#     ----------
#     X_in : ndarray, size=(n, d+1)
#         Point pattern of n samples in d dimensional euclidean space with
#         an additional mark.
#     Y_in : ndarray, size=(m, d+1)
#         Point pattern of m samples in d dimensional euclidean space with
#         an additional mark.
#     p : float
#         Power exponent parameter. Higher values penalise outliers more heavily.
#         The default is 2.
#     c : float
#         Maximum allowable localization error. This determines the error due to cardinality
#         mismatch. A lower value penalizes mismatch in cardinality less. The default is 2.

#     Returns
#     -------
#     OSPA_distance : float
#         The optimal sub-pattern assignment metric (OSPA)
#         between point patterns X_in and Y_in.
#     mark_diff : ndarray, size=(min(n, m), )
#         The difference between the mark values for the optimal sub-pattern
#         assignment.
#     KL_divergence : ndarray, size=(min(n, m), )
#         The Kullback-Leibler divergence of Y_in from X_in for
#         the optimal sub-pattern assignment.
#     """
#     # Ensure that the cardinality of X is less than that of Y.
#     if X_in.shape[0] <= Y_in.shape[0]:
#         X, Y = X_in, Y_in
#     else:
#         X, Y = Y_in, X_in
#     n, m = X.shape[0], Y.shape[0]
#     d = X.shape[1]-1

#     X_geom, Y_geom = X[:, :-1], Y[:, :-1]
#     X_mark, Y_mark = X[:, -1], Y[:, -1]

#     dist = np.linalg.norm(X_geom[:, None, :] - Y_geom[None, :, :], axis=2)
#     dist_c = np.minimum(dist, np.ones((n, m))*c)
#     bool_ = np.argmin(dist_c, axis=1)
#     sum_term = np.sum(dist_c[np.arange(n), bool_]**p)
#     card_penalty = c**p*(m-n)
#     OSPA_distance = (1/m*(sum_term + card_penalty))**(1/p)

#     mark_diff = X_mark - Y_mark[bool_]

#     if X_in.shape[0] <= Y_in.shape[0]:
#         KL_divergence = d*np.log(X_mark/Y_mark[bool_]) - d \
#                         + np.einsum("nj->n", X_geom - Y_geom[bool_])/X_mark**2 \
#                         + d*(Y_mark[bool_]/X_mark)**2
#     else:
#         KL_divergence = d*np.log(Y_mark[bool_]/X_mark) - d \
#                         + np.einsum("nj->n", X_geom - Y_geom[bool_])/Y_mark[bool_]**2 \
#                         + d*(X_mark/Y_mark[bool_])**2
#     return OSPA_distance, mark_diff, KL_divergence

def OSPA_extent(X_in, Y_in, p=2, c=2):
    """
    Computing the optimal sub-pattern assignment metric (OSPA).
    This is a metric used for quantifying the distance between point patterns
    that can potentially have different cardinalities. For reference see

    "A Consistent Metric for Performance Evaluation of Multi-Object Filters" (2008),
    By Schuhmacher, Dominic and Vo, Ba-Tuong and Vo, Ba-Ngu.

    The distance metric used for individual point-pairs in the sets is the
    Gaussian Wasserstein distance, assuming that the input has d-dimensional
    mean of the Gaussian and a mark defining the lower triangular square root
    matrix for the d-dimensional covariance of the Gaussian.

    Parameters
    ----------
    X_in : ndarray, size=(n, d+1+\dots+d)
        Point pattern of n samples in d dimensional euclidean space with
        an additional mark.
    Y_in : ndarray, size=(m, d+1+\dots+d)
        Point pattern of m samples in d dimensional euclidean space with
        an additional mark.
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

    if X_in.shape[1] == 3:
        d = 2
    elif X_in.shape[1] == 4:
        d = 3
    elif X_in.shape[1] == 5:
        d = 2
    elif X_in.shape[1] == 9:
        d = 3
    else:
        print("Something doesn't add up!")

    X_geom, Y_geom = X[:, :d], Y[:, :d]
    X_mark, Y_mark = X[:, d:], Y[:, d:]

    if X_mark.shape[1] == 1:
        X_cov = X_mark[:, 0, None, None]*np.eye(d)[None, :, :]
        Y_cov = Y_mark[:, 0, None, None]*np.eye(d)[None, :, :]
    else:
        assert X_mark.shape[1] == sum(np.arange(1, d+1)), \
            "The mark dimension must match the number of non-zero entries in the lower triangular matrix."
        tril_idx = np.tril_indices(d, k=0)
        X_cov = np.zeros((n, d, d))
        for i in range(n):
            X_lower_mat = np.zeros((d, d))
            X_lower_mat[tril_idx] = X_mark[i]
            X_cov[i] = X_lower_mat @ X_lower_mat.T
        Y_cov = np.zeros((m, d, d))
        for i in range(m):
            Y_lower_mat = np.zeros((d, d))
            Y_lower_mat[tril_idx] = Y_mark[i]
            Y_cov[i] = Y_lower_mat @ Y_lower_mat.T

    dist = GaussianWasserstein(X_geom, Y_geom, X_cov, Y_cov)
    dist_c = np.minimum(dist, np.ones((n, m))*c)
    bool_ = np.argmin(dist_c, axis=1)
    sum_term = np.sum(dist_c[np.arange(n), bool_]**p)
    card_penalty = c**p*(m-n)
    OSPA_distance = (1/m*(sum_term + card_penalty))**(1/p)
    return OSPA_distance


def GaussianWasserstein(mX, mY, cX, cY):
    """
    Parameters
    ----------
    mX : ndarray, size=(n, d)
        The d-dimensional mean for n random variables.
    mY : ndarray, size=(m, d)
        The d-dimensional mean for m random variables.
    cX : ndarray, size=(n, d, d)
        The (d x d)-dimensional covariance matrix for n random variables.
    cY : ndarray, size=(m, d, d)
        The (d x d)-dimensional covariance matrix for m random variables.

    Returns
    -------
    dist : ndarray, size=(n, m)
        The Gaussian Wasserstein distance between each pair of random variables
        in X and Y.
    """
    n, m, d = mX.shape[0], mY.shape[0], mX.shape[1]

    dist_mean = np.linalg.norm(mX[:, None, :] - mY[None, :, :], axis=2)**2 # size=(n, m)

    cov_term = np.zeros((n, m, d, d))
    sqrt_cY = np.zeros((m, d, d))
    for i in range(m):
        sqrt_cY[i] = sqrtm(cY[i])
        for j in range(n):
            cov_term[j, i] = cX[j] + cY[i] - 2*sqrtm(sqrt_cY[i] @ cX[j] @ sqrt_cY[i])
    dist_cov = np.trace(cov_term, axis1=2, axis2=3)
    dist = np.sqrt(dist_mean + dist_cov)
    return dist


if __name__ == "__main__":
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [1.5, 3.5]])
    Y = np.array([[0.9, 2.1],
                  [1.9, 2.9],
                  [2.9, 4.1]])

    GOSPA_distance = GOSPA(X, Y, c=0.5)
    print(GOSPA_distance)

    OSPA_distance = OSPA(X, Y, c=0.5)
    print(OSPA_distance)

    Tilde_X = np.array([[1, 2, 1],
                        [2, 3, 0.16],
                        [3, 4, 0.05],
                        [1.5, 3.5, 0.19]])
    Tilde_Y = np.array([[0.9, 2.1, 1.1],
                        [1.9, 2.9, 0.14],
                        [2.85, 4.1, 0.07]])
    OSPA_dist = OSPA_extent(Tilde_X, Tilde_Y, p=2, c=2)
    print(OSPA_dist)

    Tilde_X = np.array([[1, 2, 1, 0.2, 1],
                        [2, 3, 0.16, 0.05, 0.12],
                        [3, 4, 0.05, 0.01, 0.03],
                        [1.5, 3.5, 0.19, 0.1, 0.2]])
    Tilde_Y = np.array([[0.9, 2.1, 1.1, 0.5, 0.9],
                        [1.9, 2.9, 0.14, 0.1, 0.2],
                        [2.85, 4.1, 0.07, 0.02, 0.08]])
    OSPA_dist = OSPA_extent(Tilde_X, Tilde_Y, p=2, c=2)
    print(OSPA_dist)