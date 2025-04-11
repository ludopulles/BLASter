"""
Utility functions for calling Babai's nearest plane algorithm, size-reducing a basis or
Seysen-reducing a basis.

In comments, the old recursive functions are kept for clarity.
"""
from functools import cache
import numpy as np

# Local imports
from blaster_core import ZZ_matmul, FT_matmul


# Reduction properties:


def is_weakly_lll_reduced(R, delta=.99):
    """
    Return whether R is Weakly-LLL-reduced
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    n = len(R)
    for pos in range(0, n - 1):
        # vectors are b0 = (u, 0), b1 = (v, w).
        u = abs(R[pos, pos])
        v, w = R[pos, pos + 1], R[pos + 1, pos + 1]
        v_mod = ((v + u/2) % u) - u/2

        if v_mod**2 + w**2 <= delta * u**2:
            return False  # ||b1||^2 <= delta ||b0||^2
    return True


def is_size_reduced(R):
    """
    Return whether R is size-reduced.
    :param R: upper-triangular matrix
    :return: bool
    """
    return all(max(abs(R[i, i + 1:])) <= abs(R[i, i]) / 2 for i in range(len(R) - 1))


def is_lll_reduced(R, delta=.99):
    """
    Return whether R is LLL-reduced (weakly-LLL & size-reduced)
    :param R: upper-triangular matrix
    :param delta: delta-factor used in the Lovasz condition
    :return: bool
    """
    return is_weakly_lll_reduced(R, delta) and is_size_reduced(R)


@cache
def __reduction_ranges(n):
    """
    Return list of ranges that needs to be reduced.

    More generally, it returns, without using recursion, the list that would be
    the output of the following Python program:

    <<<BEGIN CODE>>>
    def rec_range(n):
        bc, res = [], []
        def F(l, r):
            if l == r:
                return
            if l + 1 == r:
                bc.append(l)
            else:
                m = (l + r) // 2
                F(l, m)
                F(m, r)
                res.append((l, m, r))
        return F(0, n)
    <<<END CODE>>>

    :param n: the length of the array that requires reduction
    :return: pair containing `the base_cases` and `result`.
             `base_cases` is a list of indices `i` such that:
                `i + 1` needs to be reduced w.r.t. `i`.
             `result` is a list of triples `(i, j, k)` such that:
                `[j:k)` needs to be reduced w.r.t. `[i:j)`.
             The guarantee is that for any 0 <= i < j < n:
             1) `i in base_cases && j = i + 1`,
             OR
             2) there is a triple (u, v, w) such that `i in [u, v)` and `j in [v, w)`.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n

            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift

            if right_idx > left_idx + 1:
                # Only consider nontrivial intervals
                if right_idx == left_idx + 2:
                    # Return length 2 intervals separately to unroll base case.
                    base_cases.append(left_idx)
                else:
                    # Properly sized interval:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))


@cache
def __babai_ranges(n):
    # Assume all indices are base cases initially
    range_around = [False] * n
    for (i, j, k) in __reduction_ranges(n)[1]:
        # Mark node `j` as responsible to reduce [i, j) wrt [j, k) once Babai is at/past index j.
        range_around[j] = (i, k)
    return range_around


# Reduction algorithms


def nearest_plane(R, T, U):
    """
    Perform Babai's Nearest Plane algorithm on multiple targets (all the columns of T), with
    respect to the upper-triangular basis R.
    This function updates T <- T + RU such that `T + RU` is in the fundamental Babai domain.
    Namely, |(T + RU)_{ij}| <= 0.5 R_ii.

    Complexity: O(N n^{omega-1}) if R is a `n x n` matrix, T is a `n x N` matrix, and `N >= n`.

    :param R: upper-triangular basis of a lattice.
    :param T: matrix containing many targets requiring reduction.
    :param U: the output transformation used to reduce T wrt R.
    :return: Nothing! The result is in T and U.
    """
    n = len(R)
    if n > 1:
        range_around = __babai_ranges(n)
        for j in range(n-1, 0, -1):
            # All targets are reduced w.r.t all basis vectors that come *after* j.
            # Compute the reduction coefficient (U_j) w.r.t basis vector j.
            U[j, :] = -np.rint((1.0 / R[j, j]) * T[j, :]).astype(np.int64)
            # Reduce jth coordinate of T wrt b_j but only in the jth coefficient!
            T[j, :] += R[j, j] * U[j, :]

            if not range_around[j]:
                T[j-1, :] += R[j-1, j] * U[j, :].astype(np.float64)
            else:
                i, k = range_around[j]
                # Apply reduction of [j:k) on the coefficients T[i:j).
                # R12, T1, U2 = R[i:j, j:k], T[i:j, :], U[j:k, :]
                # T1 = T1 + R12 · U2
                T[i:j, :] += FT_matmul(R[i:j, j:k], U[j:k, :].astype(np.float64))

    # 0 is a special case because it never needs to propagate reductions.
    U[0, :] = -np.rint((1.0 / R[0, 0]) * T[0, :]).astype(np.int64)
    # Reduce 0th coordinate of T wrt b_0 but only in the 0th coefficient!
    T[0, :] += R[0, 0] * U[0, :]


def size_reduce(R, U):
    """
    Perform size reduction on R *inplace*, and write the transformation done to R in U, such that
    calling this function with (R, U) will update the value R to R' = RU.

    Complexity: O(n^omega) for a `n x n` matrix R.

    :param R: upper-triangular basis of a lattice.
    :param U: the matrix U to store the transformation *applied* to R.
              U will be upper triangular with unit diagonal.
    :return: Nothing! R is size reduced in place.
    """
    # Assume diag(U) = (1, 1, ..., 1).
    n = len(R)

    base_cases, ranges = __reduction_ranges(n)
    for i in base_cases:
        U[i, i + 1] = -round(R[i, i + 1] / R[i, i])
        R[i, i + 1] += R[i, i] * U[i, i + 1]

    for (i, j, k) in ranges:
        # Size reduce [j, k) with respect to [i, j).
        #
        #     [R11 R12]      [U11 U12]              [S11 S12]
        # R = [ 0  R22], U = [ 0  U22], S = R · U = [ 0  S22]
        #
        # The previous iteration computed U11 and U22.
        # Currently, R11 and R22 contain the values of
        # S11 = R11 · U11 and S22 = R22 · U22 respectively.

        # W = R12 · U22
        R[i:j, j:k] = FT_matmul(R[i:j, j:k], U[j:k, j:k].astype(np.float64))

        # U12', S12 = NearestPlane(S11, W)
        nearest_plane(R[i:j, i:j], R[i:j, j:k], U[i:j, j:k])

        # U12 = U11 · U12'
        U[i:j, j:k] = ZZ_matmul(U[i:j, i:j], U[i:j, j:k])


def seysen_reduce(R, U):
    """
    Perform Seysen's reduction on a matrix R, while keeping track of the transformation matrix U.
    The matrix R is updated along the way.

    :param R: an upper-triangular matrix that will be modified
    :param U: an upper-triangular transformation matrix such that diag(U) = (1, 1, ..., 1).
    :return: Nothing! R is Seysen reduced in place.
    """
    # Assume diag(U) = (1, 1, ..., 1).
    n = len(R)

    base_cases, ranges = __reduction_ranges(n)
    for i in base_cases:
        U[i, i + 1] = -round(R[i, i + 1] / R[i, i])
        R[i, i + 1] += R[i, i] * U[i, i + 1]

    for (i, j, k) in ranges:
        # Seysen reduce [j, k) with respect to [i, j).
        #
        #     [R11 R12]      [U11 U12]              [S11 S12]
        # R = [ 0  R22], U = [ 0  U22], S = R · U = [ 0  S22]
        #
        # The previous iteration has computed U11 and U22.
        # Currently, R11 and R22 contain the values of
        # S11 = R11 · U11 and S22 = R22 · U22 respectively.

        # S12' = R12 · U22.
        R[i:j, j:k] = FT_matmul(R[i:j, j:k], U[j:k, j:k].astype(np.float64))

        # W = round(-S11^{-1} · S12').
        W = np.rint(FT_matmul(-np.linalg.inv(R[i:j, i:j]), R[i:j, j:k]))

        # U12 = U11 · W
        U[i:j, j:k] = ZZ_matmul(U[i:j, i:j], W.astype(np.int64))

        # S12 = S12' + S11 · W.
        R[i:j, j:k] += FT_matmul(R[i:j, i:j], W.astype(np.float64))


# For didactical reasons, here are the recursive versions of:
# - nearest_plane,
# - size_reduce, and
# - seysen_reduce.
#
#
# def nearest_plane(R, T, U):
#     """
#     Perform Babai's Nearest Plane algorithm on multiple targets (all the columns of T), with
#     respect to the upper-triangular basis R.
#     This function updates T <- T + RU such that `T + RU` is in the fundamental Babai domain.
#     Namely, |(T + RU)_{ij}| <= 0.5 R_ii.
#
#     Complexity: O(N n^{omega-1}) if R is a `n x n` matrix, T is a `n x N` matrix, and `N >= n`.
#
#     :param R: upper-triangular basis of a lattice.
#     :param T: matrix containing many targets requiring reduction.
#     :param U: the output transformation used to reduce T wrt R.
#     :return: Nothing! The result is in T and U.
#     """
#     n, m = R.shape[0], R.shape[0] // 2
#     if n == 1:
#         U[0, :] = -np.rint((1.0 / R[0, 0]) * T).astype(np.int64)
#         T += R[0, 0] * U
#     else:
#         # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#         # T1, T2 = T[:m, :], T[m:, :]
#         # U1, U2 = U[:m, :], U[m:, :]
#
#         # U2 = NP(R22, T2)
#         nearest_plane(R[m:, m:], T[m:, :], U[m:, :])
#
#         # T1 = T1 + R12 · U2
#         T[:m, :] += FT_matmul(R[:m, m:], U[m:, :].astype(np.float64))
#
#         # U1 = NP(R11, T1)
#         nearest_plane(R[:m, :m], T[:m, :], U[:m, :])
#
#
# def size_reduce(R, U):
#     """
#     Perform size reduction on R *inplace*, and write the transformation done to R in U, such that
#     calling this function with (R, U) will update the value R to R' = RU.
#
#     Complexity: O(n^omega) for a `n x n` matrix R.
#
#     :param R: upper-triangular basis of a lattice.
#     :param U: the matrix U to store the transformation *applied* to R.
#               U will be upper triangular with unit diagonal.
#     :return: Nothing! R is size reduced in place.
#     """
#     n, m = R.shape[0], R.shape[0] // 2
#     if n == 1:
#         return
#
#     if n == 2:
#         U[0, 1] = -round(R[0, 1] / R[0, 0])
#         R[0, 1] += R[0, 0] * U[0, 1]
#     else:
#         # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#         # U11, U12, U22 = U[:m, :m], U[:m, m:], U[m:, m:]
#
#         # U11 = SizeReduce(R11)
#         size_reduce(R[:m, :m], U[:m, :m])
#
#         # U22 = SizeReduce(R22)
#         size_reduce(R[m:, m:], U[m:, m:])
#
#         # R12 = R12 · U22
#         R[:m, m:] = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#         # U12' = NearestPlane(basis=R11', target=R12), R12 = R12 + R11' U12'
#         nearest_plane(R[:m, :m], R[:m, m:], U[:m, m:])
#
#         # Note: NP was called with the size-reduced R11' = R11 · U11.
#         # U12 = U11 · U12'
#         # U[:m, m:] = U[:m, :m] @ U[:m, m:]
#         U[:m, m:] = ZZ_matmul(U[:m, :m], U[:m, m:])
#
#
# def seysen_reduce(R, U):
#    """
#    Seysen reduce a matrix R, recursive style, and store the result in U.
#    See: Algorithm 7 from [KEF21].
#    [KEF21] P. Kircher, T. Espitau, P.-A. Fouque. Towards faster polynomial-time lattice reduction.
#    :param R: an upper-triangular matrix (having row vectors).
#    :param U: a unimodular transformation U such that RU is Seysen-Reduced.
#    :return: None! The result is stored in U.
#    """
#    n, m = len(R), len(R) // 2
#    if n == 1:
#        # Base case
#        U[0, 0] = 1
#    elif n == 2:
#        # Make sure RU is size-reduced, i.e. |R00*X + R01| <= |R00|/2
#        U[0, 0] = U[1, 1] = 1
#        U[0, 1] = -round(R[0, 1] / R[0, 0])
#    else:
#        # R11, R12, R22 = R[:m, :m], R[:m, m:], R[m:, m:]
#        seysen_reduce(R[:m, :m], U[:m, :m])
#        seysen_reduce(R[m:, m:], U[m:, m:])
#
#        # S11 = R11 · U11
#        S11 = FT_matmul(R[:m, :m], U[:m, :m].astype(np.float64))
#
#        # S12' = R12 · U22
#        S12 = FT_matmul(R[:m, m:], U[m:, m:].astype(np.float64))
#
#        # W = round(-S11^{-1} S12').
#        W = np.rint(FT_matmul(-np.linalg.inv(S11), S12)).astype(np.int64)
#
#        # U12 = U11 · W
#        U[:m, m:] = ZZ_matmul(U[:m, :m], W)
