"""
Utility functions for computing the basis profile of a given basis, and measuring its quality.
"""
from math import exp, gamma, log2, pi
import numpy as np


def get_profile(basis, is_upper=False):
    """
    Return the profile of a basis, i.e. log_2 ||b_i*|| for i=1, ..., n.
    Note: the logarithm is done base 2, similar to https://github.com/keeganryan/flatter.
    :param basis: basis for a lattice
    :param is_upper: whether `basis` is already an upper triangular matrix or not
    """
    upper = basis if is_upper else np.linalg.qr(basis, mode='r')
    return [log2(abs(d_i)) for d_i in upper.diagonal()]


def gh(dim):
    """
    Return the Gaussian Heuristic at dimension n. This gives a prediction of
    the length of the shortest vector in a lattice of unit volume.
    :param n: lattice dimension
    :return: GH(n)
    """
    if dim >= 100:
        return float(dim / (2*pi*exp(1)))**0.5
    return float(gamma(1.0 + 0.5 * dim)**(1.0 / dim) / pi**0.5)


def gaussian_heuristic(basis):
    """
    Return the Gaussian Heuristic for a particular basis.
    """
    rank = basis.shape[1]
    return gh(rank) * 2.0**(sum(get_profile(basis)) / rank)


def rhf(profile):
    """
    Return the n-th root Hermite factor, given the profile of some basis, i.e.
        rhf(B) = (||b_0|| / det(B)^{1/n})^{1/n}.
    :param profile: profile belonging to some basis of some lattice
    """
    n = len(profile)
    return 2.0**((profile[0] - sum(profile) / n) / n)


def slope(profile):
    """
    Return the current slope of a profile
    """
    n = len(profile)
    i_mean = (n - 1) * 0.5
    v1 = sum(profile[i] * (i - i_mean) for i in range(n))
    v2 = sum((i - i_mean)**2 for i in range(n))
    return v1 / v2


def potential(profile):
    """
    Return the (log2 of the) potential of a basis profile.
    Normally in lattice reduction, this is a strictly decreasing function of time, and is used to
    prove that LLL runs in polynomial time.
    """
    n = len(profile)
    return sum((n - i) * profile[i] for i in range(n))
