# distutils: language = c++

import numpy as np
cimport cython
cimport numpy as cnp

from cysignals.signals cimport sig_on, sig_off
from cython.parallel cimport prange
from libc.string cimport memcpy
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num

from decl cimport FT, ZZ, lll_reduce, deeplll_reduce, bkz_reduce


cnp.import_array()  # http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types
NP_FT = np.float64  # floating-point type
NP_ZZ = np.int64  # integer type


cdef int debug_size_reduction = 0


def set_debug_flag(int flag):
    global debug_size_reduction
    debug_size_reduction = flag


def set_num_cores(int num_cores):
    omp_set_num_threads(num_cores)  # used by `prange` in block_X


# Lattice reduction
@cython.boundscheck(False)
@cython.wraparound(False)
def block_lll(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run LLL on block [i, i + w).
        lll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])


@cython.boundscheck(False)
@cython.wraparound(False)
def block_deep_lll(int depth,
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run deep-LLL on block [i, i + w).
        deeplll_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, depth)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])


@cython.boundscheck(False)
@cython.wraparound(False)
def block_bkz(int beta,
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef:
        Py_ssize_t n = R.shape[0]
        int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
        FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
        ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

    # Check that these are of the correct type:
    assert R.dtype == NP_FT and U.dtype == NP_ZZ

    sig_on()
    for block_id in prange(num_blocks, nogil=True):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        for j in range(w):
            memcpy(&R_sub[block_id, j * w], &R[i + j, i], w * sizeof(FT));

        # Step 1: run BKZ on block [i, i + w).
        bkz_reduce(w, &R_sub[block_id, 0], &U_sub[block_id, 0], delta, beta)

        if debug_size_reduction != 0:
            for j in range(w):
                memcpy(&R[i + j, i], &R_sub[block_id, j * w], w * sizeof(FT));

    sig_off()

    # Step 2: Update U and B_red locally by multiplying with U_sub[block_id].
    for block_id in range(num_blocks):
        i = offset + block_size * block_id
        w = min(n - i, block_size)

        ZZ_right_matmul_strided(U[:, i:i+w], U_sub[block_id, 0:w*w])
        ZZ_right_matmul_strided(B_red[:, i:i+w], U_sub[block_id, 0:w*w])


# Matrix Multiplication
# ZZ (integer type)
@cython.boundscheck(False)
@cython.wraparound(False)
def ZZ_matmul(const ZZ[:, :] A, const ZZ[:, :] B) -> cnp.ndarray[ZZ]:
    cdef:
        int n = A.shape[0], m = A.shape[1], k = B.shape[1], a, b, c
        ZZ[:, ::1] C = np.zeros(shape=(n, k), dtype=NP_ZZ)

    assert B.shape[0] == m, "Dimension mismatch"

    for a in prange(n, nogil=True):
        for c in range(k):
            for b in range(m):
                C[a, c] += A[a, b] * B[b, c]
    return np.asarray(C)


# Variant with row stride for A:
# Note: B is a 1D-array of length m^2. This is used in lattice_reduction.pyx
@cython.boundscheck(False)
@cython.wraparound(False)
def ZZ_right_matmul_strided(ZZ[:, :] A, const ZZ[:] B) -> None:
    cdef:
        int n = A.shape[0], m = A.shape[1], stride_a = A.strides[0] // sizeof(ZZ), a, b, c
        ZZ[:, ::1] C = np.zeros(shape=(n, m), dtype=NP_ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"
    for a in prange(n, nogil=True):
        for c in range(m):
            for b in range(m):
                C[a, c] += A[a, b] * B[b * m + c]
    for a in prange(n, nogil=True):
        for c in range(m):
            A[a, c] = C[a, c]


# FT (floating-point type)
@cython.boundscheck(False)
@cython.wraparound(False)
def FT_matmul(FT[:, :] A, FT[:, :] B) -> cnp.ndarray[FT]:
    cdef:
        int n = A.shape[0], m = A.shape[1], k = B.shape[1], a, b, c
        FT[:, ::1] C = np.zeros(shape=(n, k), dtype=NP_FT)

    assert B.shape[0] == m, "Dimension mismatch"

    for a in prange(n, nogil=True):
        for c in range(k):
            for b in range(m):
                C[a, c] += A[a, b] * B[b, c]

    return np.asarray(C)
