#include <Eigen/Core>
#include <Eigen/Dense>

#include "types.hpp"


using Matrix = Eigen::Matrix<ZZ, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
typedef Eigen::Stride<Eigen::Dynamic, 1> Stride;


void eigen_init(int num_cores) {
	// See https://eigen.tuxfamily.org/dox/TopicMultiThreading.html
	Eigen::setNbThreads(num_cores);
}


// See: https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `c`.
 * Dimensions of `a`, `b` and `c` are assumed to be `n x m`, `m x k` and `n x k` respectively.
 */
void eigen_matmul(const ZZ *a, const ZZ *b, ZZ *c, int n, int m, int k) {
	Eigen::Map<const Matrix> ma(a, n, m), mb(b, m, k);
	Eigen::Map<Matrix> mc(c, n, k);

	mc = ma * mb;
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `b`.
 * Dimensions of `a` and `b` are assumed to be `n x n` and `n x m` respectively.
 */
void eigen_left_matmul(const ZZ *a, ZZ *b, int n, int m, int stride_a, int stride_b) {
	Eigen::Map<const Matrix, Eigen::Unaligned, Stride> ma(a, n, n, Stride(stride_a, 1));
	Eigen::Map<Matrix, Eigen::Unaligned, Stride> mb(b, n, m, Stride(stride_b, 1));

	mb = ma * mb;
}

/**
 * Compute the matrix product between a and b, and store the result `a * b` in `a`.
 * Dimensions of `a` and `b` are assumed to be `n x m` and `m x m` respectively.
 */
void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m) {
	Eigen::Map<Matrix> ma(a, n, m);
	Eigen::Map<const Matrix> mb(b, m, m);

	ma *= mb;
}

void eigen_right_matmul(ZZ *a, const ZZ *b, int n, int m, int stride_a) {
	Eigen::Map<Matrix, Eigen::Unaligned, Stride> ma(a, n, m, Stride(stride_a, 1));
	Eigen::Map<const Matrix> mb(b, m, m);

	ma *= mb;
}
