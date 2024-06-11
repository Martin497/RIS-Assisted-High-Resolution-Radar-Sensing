# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:41:14 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Base functionality to do data association. (13/02/2024)
"""


import numpy as np
from scipy.optimize import linear_sum_assignment


def data_association_cost(phiN, phiR):
    """
    Compute the cost matrix for the data association problem.
    """
    n, m = phiN.shape[0], phiR.shape[0]
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for k in range(m):
            term1 = min(np.abs(phiN[i, 0]-phiR[k, 0]), np.abs(np.abs(phiN[i, 0]-phiR[k, 0])-2*np.pi))**2
            term2 = min(np.abs(phiN[i, 1]-phiR[k, 1]), np.abs(np.abs(phiN[i, 1]-phiR[k, 1])-np.pi))**2
            # term1 = (np.abs(phiN[n, 0]-phiR[m, 0]) % (2*np.pi))**2
            # term2 = (np.abs(phiN[n, 1]-phiR[m, 1]) % (np.pi))**2
            cost_matrix[i, k] = term1 + term2
    return cost_matrix

def murty(cost_matrix):
    """
    Solve the linear sum assignment problem for (n\times m)-dimensional
    real cost_matrix.
    """
    n, m = cost_matrix.shape
    assignments = np.zeros(n, dtype=np.int8)

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    assignments = col_idx
    cost = cost_matrix[row_idx, col_idx].sum()
    return assignments, cost

def data_association(cost_matrix, square=False):
    """
    Solve the non-square data association problem.

    Input:
    ------
        cost_matrix : ndarray, size=(n, m)

    Output:
    -------
        If m == n:
            assignments : ndarray, size=(n,), dtype=np.int8
                Array of integers assigning a column to each row.
            cost: float
                The aggregated cost of the assignment.
        If m > n:
            assignments : list of length n of lists of integers
                The i-th entry in the list is a list of integers
                associating one or more columns to the i-th row.
            cost : float
                The aggregated cost of the assignment.
    """
    n, m = cost_matrix.shape
    assert m >= n, """The implementation of data association assumes that
    the number of columns is greater than or equal to the number of rows. To use the method, transpose the problem."""

    if m == n:
        assignments, cost = murty(cost_matrix)
        return assignments, cost
    elif m > n:
        init_assignments, cost = murty(cost_matrix)
        if square is True:
            return init_assignments, cost
        assignments = [[a] for a in init_assignments]
        aran = np.arange(m)
        # init_assignments_sorted = np.sort(init_assignments)
        # print(init_assignments_sorted)
        # print(np.searchsorted(init_assignments_sorted,aran))
        # print(init_assignments_sorted[np.searchsorted(init_assignments_sorted,aran)] !=  aran)
        # non_assigned = aran[init_assignments_sorted[np.searchsorted(init_assignments_sorted,aran)] !=  aran]
        non_assigned = np.array([a for a in aran if a not in init_assignments])
        filler_assignments = np.argmin(cost_matrix[:, non_assigned], axis=0)
        for idx1, idx2 in zip(non_assigned, filler_assignments):
            assignments[idx2] = assignments[idx2] + [idx1]
            cost += cost_matrix[idx2, idx1]
        return assignments, cost

