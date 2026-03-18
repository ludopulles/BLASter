#include<algorithm> // std::swap, std::fill_n
#include<cmath> // llround, sqrt

#include "enumeration.cpp"
#include "pruning_params.cpp"


/*******************************************************************************
 * Helper functions to access the matrices R and U at row 'row' and column 'col'
 ******************************************************************************/
#define RR(row, col) R[(row) * N + (col)]
#define RSQ(row, col) (RR(row, col) * RR(row, col))

#define UU(row, col) U[(row) * N + (col)]

/*
 * Replace `b_j` by `b_j + number * b_i`, and
 * update R-factor and transformation matrix U accordingly.
 * Assumes i < j.
 *
 * Complexity: O(N)
 */
inline void alter_basis(const int N, FT *R, ZZ *U, int i, int j, ZZ number)
{
	if (number == 0) {
		return;
	}

	// R_j += number * R_i.
	for (int k = 0; k <= i; k++) {
		RR(k, j) += number * RR(k, i);
	}

	// U_i += number * U_i.
	for (int k = 0; k < N; k++) {
		UU(k, j) += number * UU(k, i);
	}
}

/*
 * Size reduce column j with respect to column i (i < j), and
 * update the R-factor and transformation matrix U accordingly.
 *
 * Complexity: O(N)
 */
inline void size_reduce(const int N, FT *R, ZZ *U, int i, int j)
{
	alter_basis(N, R, U, i, j, llround(-RR(i, j) / RR(i, i)));
}

/*
 * Swap the adjacent basis vectors b_k and b_{k+1} and update the R-factor and transformation
 * matrix U correspondingly. R is updated by performing a Givens rotation.
 *
 * Complexity: O(N)
 */
void swap_basis_vectors(const int N, FT *R, ZZ *U, const int k)
{
	// a. Perform Givens rotation on coordinates {k, k+1}, and update R.
	FT c = RR(k, k + 1), s = RR(k + 1, k + 1), norm = sqrt(c * c + s * s);
	c /= norm;
	s /= norm;

	RR(k, k + 1) = c * RR(k, k);
	RR(k + 1, k + 1) = s * RR(k, k);
	RR(k, k) = norm;

	for (int i = k + 2; i < N; i++) {
		FT new_value = c * RR(k, i) + s * RR(k + 1, i);
		RR(k + 1, i) = s * RR(k, i) - c * RR(k + 1, i);
		RR(k, i) = new_value;
	}

	// b. Swap R_k and R_{k+1}, except the already processed 2x2 block.
	for (int i = 0; i < k; i++) {
		std::swap(RR(i, k), RR(i, k + 1));
	}

	// c. Swap U_k and U_{k+1}.
	for (int i = 0; i < N; i++) {
		std::swap(UU(i, k), UU(i, k + 1));
	}
}

inline void init_U(const int N, ZZ *U)
{
	// Initialize U with the identity matrix
	std::fill_n(U, N * N, ZZ(0));
	for (int i = 0; i < N; i++) {
		UU(i, i) = 1;
	}
}

/*******************************************************************************
 * LLL reduction
 ******************************************************************************/
void _lll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int limit_k)
{
	// Loop invariant: [0, k) is LLL-reduced (size reduced and Lovász' condition holds).
	for (int k = 1; k < limit_k; ) {
		// 1. Size-reduce b_k wrt b_0, ..., b_{k-1}.
		for (int i = k - 1; i >= 0; --i) {
			size_reduce(N, R, U, i, k);
		}

		// 2. Check Lovász’ condition: `\delta ||\pi(b_{k-1})||^2 <= ||\pi(b_k)||^2`.
		if (delta * RSQ(k - 1, k - 1) <= RSQ(k - 1, k) + RSQ(k, k)) {
			// Lovász’ condition is satisfied at `k`, so increment `k`.
			k++;
		} else {
			// Lovász’ condition is not satisfied at `k`.
			// 3. Swap b_{k - 1} and b_k.
			swap_basis_vectors(N, R, U, k - 1);

			// 4. Decrease `k` if possible.
			if (k > 1) k--;
		}
	}
}

/*
 * Perform LLL reduction on the basis R, and return the transformation matrix U such that RU is
 * LLL-reduced.
 *
 * @param R upper-triangular matrix representing the R-factor from QR decomposing the basis.
 * @param U transformation matrix, that was applied to R to LLL-reduce it.
 *
 * Complexity: poly(N) (for a fixed delta < 1).
 */
void lll_reduce(const int N, FT *R, ZZ *U, const FT delta)
{
	init_U(N, U);
	_lll_reduce(N, R, U, delta, N);
}

/*******************************************************************************
 * LLL reduction with deep insertions
 ******************************************************************************/
void _deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth, const int limit_k)
{
	// First run LLL, because this makes deep-LLL faster, see [2].
	// [2] https://doi.org/10.1007/s10623-014-9918-8
	_lll_reduce(N, R, U, delta, limit_k);

	// Loop invariant: [0, k) is depth-deep-LLL-reduced.
	for (int k = 1; k < limit_k; ) {
		// 1. Size-reduce R_k wrt R_0, ..., R_{k - 1}.
		for (int i = k - 1; i >= 0; i--) {
			size_reduce(N, R, U, i, k);
		}

		// 2. Determine ||b_k||^2
		FT proj_norm_sq = 0.0;
		for (int i = 0; i <= k; i++) {
			proj_norm_sq += RSQ(i, k);
		}

		// 3. Look for an i < k such that ||pi_i(b_k)||^2 <= delta ||b_i*||^2.
		// Loop invariant: proj_norm_sq = ||pi_i(b_k)||^2.
		bool swap_performed = false;
		for (int i = 0; i < k; i++) {
			if ((i < depth || i >= k - depth) && proj_norm_sq <= delta * RSQ(i, i)) {
				// 3a. Deep insert b_k at position i and move b_i, ..., b_{k-1}
				// one position forward. Complexity: O(N * (k - i))
				while (k > i) {
					swap_basis_vectors(N, R, U, --k);
				}
				if (k == 0) k++;
				swap_performed = true;
				break;
			}

			// 3b. Increment i and update ||pi_i(b_k)||^2.
			proj_norm_sq -= RSQ(i, k);
		}

		if (!swap_performed) k++;
	}
}

/*
 * Perform depth-deep-LLL reduction on the basis R, and return the transformation matrix U such
 * that RU is depth-deep-LLL-reduced.
 *
 * @param R upper-triangular matrix representing the R-factor from QR decomposing the basis.
 * @param U transformation matrix, that was applied to R to deep-LLL-reduce it.
 * @param depth maximum number of positions allowed for 'deep insertions'
 *
 * Complexity: poly(N) (for a fixed delta < 1 and a fixed depth).
 */
void deeplll_reduce(const int N, FT *R, ZZ *U, const FT delta, const int depth)
{
	 init_U(N, U);
	 _deeplll_reduce(N, R, U, delta, depth, N);
}

/*******************************************************************************
 * BKZ reduction
 ******************************************************************************/

/*
 * Compute and return the square of the Gaussian Heuristic, i.e. (the square
 * of) the expected length of the shortest nonzero vector, for a lattice of
 * dimension `dimension` with a determinant (covolume) of `exp(log_volume)`.
 *
 * @param dimension dimension of lattice
 * @param log_volume logarithm of volume of the lattice, natural base.
 * @return GH(L)^2, for a lattice L of rank `dimension` and `det(L) = exp(log_volume)`.
 */
FT gh_squared(int dimension, FT log_volume)
{
	// GH(n) = Gamma(n / 2 + 1) / (pi)^{n / 2} so
	// GH(n)^2 = exp(log(Gamma(n / 2 + 1)) / n) / pi.
	FT log_gamma = lgamma(dimension / 2.0 + 1);
	return exp(2.0 * (log_gamma + log_volume) / dimension) / M_PI;
}

FT safe_gh_squared(int dim, FT log_volume)
{
	// Loosely based on Figure 2 from [3]:
	FT gh2 = gh_squared(dim, log_volume);
	FT gh_factor = std::max(1.05, std::min(2.0, 1.0 + 4.0 / dim));
	return gh2 * gh_factor * gh_factor;
}

/*
 * Solve SVP on b_[0, N), and
 * put the result somewhere in the basis where the coefficient is +1/-1, and
 * run LLL on b_0, ..., b_{N-1}.
 *
 * Based on Algorithm 1 from [3].
 * [3] https://doi.org/10.1007/978-3-642-25385-0_1
 */
void svp(const int N, FT *R, ZZ *U, const FT delta, int i, int w, ZZ *sol)
{
	// Solve SVP on block [i, i + w).
	FT log_volume = 0.0;
	for (int j = 0; j < w; j++) {
		log_volume += log(RSQ(i + j, i + j));
	}

	// Find a solution that is shorter than current basis vector and (1 + eps)·GH
	FT expected_normsq = std::min((1023.0 / 1024) * RSQ(i, i), safe_gh_squared(w, log_volume));

	// Pick the pruning parameters for `pr[0 ... w - 1]`.
	const FT *pr = get_pruning_coefficients(w);

	// [3] Algorithm 1, line 4:
	// Perform enumeration to find the shortest vector.
	FT sol_square_norm = enumeration(w, &RR(i, i), N, pr, expected_normsq, sol);

	// [3] Algorithm 1, line 5:
	// Check if a shorter, nonzero vector is found.
	if (sol_square_norm > 0.0) {
		// Find the last nonzero coefficient.
		int j = w - 1;
		while (j > 0 && sol[j] == 0) {
			j--;
		}

		// Replace `v` by `-v`, if sol[j] = -1.
		if (j > 0 && sol[j] == -1) {
			for (int k = 0; k <= j; k++) {
				sol[k] = -sol[k];
			}
		}

		// Only do an insertion of the shorter vector if sol[j] = 1.
		if (j > 0 && sol[j] == 1) {
			// Update `b_{i + j} <-- \sum_{k=0}^j sol[k] b_{i + k}`.
			for (int k = 0; k < j; k++) {
				// for all 0 <= k < j: b_{i + j} += sol[k] * b_{i + k}.
				alter_basis(N, R, U, i + k, i + j, sol[k]);
			}

			// Move b_{i + j} to position b_i.
			while (--j >= 0) {
				swap_basis_vectors(N, R, U, i + j);
			}
		}
	}

	/* There are three possible reasons why no update was performed:
	 * 1. A shorter vector is not found because of pruning
	 * 2. `b_i` is already the shortest vector in the block [i, i + w)
	 * 3. The solution coefficients do not allow an easy insertion.
	 *
	 * Note 1: See pruning_params.cpp for the success probability.
	 * Note 2: In practice, reason 3 seldomly happens, when calling progressive BKZ. The algorithm
	 *         could be modified to handle such difficult insertions.
	 */

	// [3] Algorithm 1, line 6 or 8
	// DeepLLL-reduce [0, i + w) such that the next enumeration runs on a DeepLLL-reduced basis.
	_deeplll_reduce(N, R, U, delta, 4, std::min(i + w, N));
}

/*
 * Perform BKZ-beta reduction on the basis R, and return the transformation matrix U such that
 * RU is BKZ-beta-reduced.
 *
 * @param R upper-triangular matrix representing the R-factor from QR decomposing the basis.
 * @param U transformation matrix, that was applied to R to deep-LLL-reduce it.
 * @param beta block size used for BKZ (dimension of SVP oracle that uses enumeration).
 *
 * Complexity: poly(N) * beta^{c_BKZ beta} for a fixed delta < 1, where c_BKZ ~ 0.125 in [1].
 * [1] https://doi.org/10.1007/978-3-030-56880-1_7
 */
void bkz_reduce(const int N, FT *R, ZZ *U, const FT delta, int beta)
{
	ZZ sol[MAX_ENUM_N]; // coefficients of the enumeration solution for SVP in block of size beta.

	// First init U and run deep-LLL, before performing BKZ.
	deeplll_reduce(N, R, U, delta, 4);

	if (beta <= 2) return;

	if (beta > N) {
		// Perform one HKZ-tour.
		// Note: this is only done at the end of the global basis!
		for (int i = 0; i + 2 <= N; i++) {
			// Solve SVP on block [i, N).
			svp(N, R, U, delta, i, N - i, sol);
		}
	} else {
		// Perform one BKZ-tour.
		for (int i = 0; i + beta <= N; i++) {
			// Solve SVP on block [i, i + beta).
			svp(N, R, U, delta, i, beta, sol);
		}
	}
}
