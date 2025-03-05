# distutils: language = c++

cdef extern from "types.hpp":
    int MAX_ENUM_N
    ctypedef double FT # floating-point type
    ctypedef long long ZZ # integer type

cdef extern from "block_lll.cpp":
    void lll_reduce(const int N, FT *R, ZZ *U, const FT delta) noexcept nogil
    void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth) noexcept nogil
    void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, const int beta) noexcept nogil

cdef extern from "eigen_matmul.cpp":
    void eigen_init(int num_cores) noexcept nogil

	# c = a * b
    void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) noexcept nogil
    void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k, int stride_a) noexcept nogil

	# b = a * b
    void eigen_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b) noexcept nogil

	# a = a * b
    void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m) noexcept nogil
    void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m, int stride_a) noexcept nogil


cdef extern from "pruning_params.cpp":
    FT* get_pruning_coefficients(int block_size) noexcept nogil

cdef extern from "enumeration.cpp":
    FT enumeration(int N, FT *R, int rowstride, FT *pruningvector, FT enumeration_radius, ZZ* sol) noexcept nogil
    FT enumeration_lastone(int N, FT *R, int rowstride, FT *pruningvector, ZZ* sol) noexcept nogil
