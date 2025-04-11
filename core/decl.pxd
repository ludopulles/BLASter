# distutils: language = c++

cdef extern from "types.hpp":
    int MAX_ENUM_N
    ctypedef double FT # floating-point type
    ctypedef long long ZZ # integer type

cdef extern from "block_lll.cpp":
    void lll_reduce(const int N, FT *R, ZZ *U, const FT delta) noexcept nogil
    void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth) noexcept nogil
    void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, const int beta) noexcept nogil
