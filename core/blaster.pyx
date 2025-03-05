# distutils: language = c++

import numpy as np
cimport cython
cimport numpy as cnp

from cysignals.signals cimport sig_on, sig_off
from cython.parallel cimport prange
from libc.string cimport memcpy
from openmp cimport omp_set_num_threads, omp_get_num_threads, omp_get_thread_num

from decl cimport FT, ZZ, \
    lll_reduce, deeplll_reduce, bkz_reduce, \
    eigen_init, eigen_matmul, eigen_left_matmul, eigen_right_matmul, \
    get_pruning_coefficients, enumeration, enumeration_lastone


cnp.import_array()  # http://docs.cython.org/en/latest/src/tutorial/numpy.html#adding-types
NP_FT = np.float64  # floating-point type
NP_ZZ = np.int64  # integer type


cdef int debug_size_reduction = 0


def set_debug_flag(int flag):
    global debug_size_reduction
    debug_size_reduction = flag


def set_num_cores(int num_cores):
    omp_set_num_threads(num_cores)  # used by `prange` in block_X
    eigen_init(num_cores)


# Lattice reduction
@cython.boundscheck(False)
@cython.wraparound(False)
def block_lll(
        cnp.ndarray[FT, ndim=2] R, cnp.ndarray[ZZ, ndim=2] B_red, cnp.ndarray[ZZ, ndim=2] U,
        FT delta, int offset, int block_size) -> None:
    global debug_size_reduction

    # Variables
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

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
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

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
    cdef Py_ssize_t n = R.shape[0]
    cdef int i, j, w, num_blocks = int((n - offset + block_size - 1) / block_size), block_id
    cdef FT[:, ::1] R_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_FT)
    cdef ZZ[:, ::1] U_sub = np.empty(shape=(num_blocks, block_size**2), dtype=NP_ZZ)

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
def ZZ_matmul(const ZZ[:, ::1] A, const ZZ[:, ::1] B) -> cnp.ndarray[ZZ]:
    cdef int n = A.shape[0], m = A.shape[1], k = B.shape[1]
    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    assert B.shape[0] == m, "Dimension mismatch"
    eigen_matmul(<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k)
    return np.asarray(C)


# Variant with row stride for A:
def ZZ_matmul_strided(const ZZ[:, :] A, const ZZ[:, ::1] B) -> cnp.ndarray[ZZ]:
    cdef int n = A.shape[0], m = A.shape[1], k = B.shape[1]
    cdef int stride_a = A.strides[0] // sizeof(ZZ)

    assert B.shape[0] == m, "Dimension mismatch"
    assert A.strides[1] == sizeof(FT), "Array A is not C-contiguous"

    cdef ZZ[:, ::1] C = np.empty(shape=(n, k), dtype=NP_ZZ)

    eigen_matmul(<const ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], &C[0, 0], n, m, k, stride_a)
    return np.asarray(C)


def ZZ_left_matmul_strided(const ZZ[:, :] A, ZZ[:, :] B) -> None:
    cdef int n = B.shape[0], m = B.shape[1]
    cdef int stride_a = A.strides[0] // sizeof(ZZ), stride_b = B.strides[0] // sizeof(ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"
    assert B.strides[1] == sizeof(ZZ), "Array B not C-contiguous"

    eigen_left_matmul(<const ZZ*>&A[0, 0], <ZZ*>&B[0, 0], n, m, stride_a, stride_b)


def ZZ_right_matmul(ZZ[:, ::1] A, const ZZ[:, ::1] B) -> None:
    cdef int n = A.shape[0], m = A.shape[1]

    assert B.shape[0] == m and B.shape[1] == m, "Dimension mismatch"

    eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0, 0], n, m)


# Variant with row stride for A:
# Note: B is a 1D-array of length m^2. This is used in lattice_reduction.pyx
def ZZ_right_matmul_strided(ZZ[:, :] A, const ZZ[:] B) -> None:
    cdef int n = A.shape[0], m = A.shape[1]
    cdef int stride_a = A.strides[0] // sizeof(ZZ)

    assert A.strides[1] == sizeof(ZZ), "Array A not C-contiguous"

    eigen_right_matmul(<ZZ*>&A[0, 0], <const ZZ*>&B[0], n, m, stride_a)

# FT (floating-point type)
def FT_matmul(cnp.ndarray[FT, ndim=2] A, cnp.ndarray[FT, ndim=2] B) -> cnp.ndarray[FT]:
    # Note: NumPy uses BLAS to multiply floating-point matrices, but Eigen uses OpenMP
    return A @ B


def enumerate_svp(const FT[:, :] R) -> tuple(FT, ZZ[::1]):
    """
    Retrieve the shortest vector in `R` expressed as `R * sol` with `sol` integral,
    such that its square norm is shorter than `R[0, 0]^2`.
    :return: the pair `(|| R * sol ||^2, sol)`.
    """
    # Variables
    cdef Py_ssize_t n = R.shape[0], stride = R.strides[0] // sizeof(ZZ)
    cdef ZZ[::1] sol = np.empty(shape=(n), dtype=NP_ZZ)

    # Get default pruning parameters
    cdef const FT *pr = get_pruning_coefficients(n)

    cdef FT norm_square = enumeration(n, &R[0, 0], stride, pr, R[0, 0]**2, <ZZ*>&sol[0])
    return norm_square, np.asarray(sol)

def enumerate_lastone(const FT[:, :] R) -> tuple(FT, ZZ[::1]):
    """
    Warning: this is a hack.

    Retrieve the shortest vector in `R` expressed as `R * sol` such that `sol[-1] = 1`.
    :return: the pair `(|| R * sol ||^2, sol)`.
    """
    # Variables
    cdef Py_ssize_t n = R.shape[0], stride = R.strides[0] // sizeof(ZZ)
    cdef ZZ[::1] sol = np.empty(shape=(n), dtype=NP_ZZ)

    # Get default pruning parameters, might be optimistic?
    cdef const FT *pr = get_pruning_coefficients(n)

    cdef FT norm_square = enumeration_lastone(n, &R[0, 0], stride, pr, <ZZ*>&sol[0])
    return norm_square, np.asarray(sol)
