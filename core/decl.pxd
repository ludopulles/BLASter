# distutils: language = c++

cdef extern from "types.hpp":
    int MAX_ENUM_N
    ctypedef double FT # floating-point type
    ctypedef long long ZZ # integer type

cdef extern from "lattice_reduction.cpp" nogil:
    void lll_reduce(const int N, FT *R, ZZ *U, const FT delta)
    void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth)
    void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, const int beta)

cdef extern from "eigen_matmul.cpp" nogil:
    void eigen_init(int num_cores)

    # c = a * b
    void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k)

    # b = a * b
    void eigen_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b)

    # a = a * b
    void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m)
    void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m, int stride_a)
