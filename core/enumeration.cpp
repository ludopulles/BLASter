#ifndef ENUMLIB_WRAPPER_ENUMERATION_CPP
#define ENUMLIB_WRAPPER_ENUMERATION_CPP

#include <iostream>

#include "enumeration.hpp"

/*
 * Perform enumeration to solve SVP in dimension N, using the enumlib library by Marc Stevens.
 *
 * @param N is dimension
 * @param R: upper-diagonal matrix of dimension N*N. B=Q*R
 * @param rowstride: rowstride of R. R(row,col) = R[rowstride*row + col]
 * @param pruningvector: vector of dimension N containing *relative* bounds for the squared norm within the projected sublattices.
 * @param enumeration_radius: expected norm squared of the shortest nonzero vector in this lattice
 * @param sol: return param: integer vector solution with respect to current basis, or the 0 vector otherwise
 *
 * Complexity: exponential in N.
 */
FT enumeration(const int N, const FT *R, const int rowstride, const FT *pruningvector, FT enumeration_radius, ZZ* sol)
{
    // ensure we always return the 0-vector in sol, unless a valid solution is found
    std::fill(sol, sol+N, ZZ(0));

    // enumeration is only supported up to MAX_ENUM_N dimensions
    if (N > MAX_ENUM_N || N <= 0)
        return 0.0;

    // we pad the enumeration tree with virtual basis vectors up to dim MAX_ENUM_N
    // these virtual basis vectors have very high length
    // thus these will have zero coefficients in any found solution
    lattice_enum_t<MAX_ENUM_N> enumobj;

    // initialize enumobj.muT
    // assumption: enumobj.muT is all-zero
    for (int i = 0; i < N-1; ++i)
    {
        FT* muTi = &enumobj.muT[i][0];
        const FT* Ri = R+(i*rowstride);
        FT Rii_inv = FT(1.0) / Ri[i];
        for (int j = i+1; j < N; ++j)
        {
            // muT[i][j] = <bj,bi*> / ||bi*||^2
            muTi[j] = Ri[j] * Rii_inv;
        }
    }

    // initialize enumobj.risq
    for (int i = 0; i < N; ++i)
    {
        // risq[i] = ||bi*||^2
        enumobj.risq[i] = R[i*rowstride+i] * R[i*rowstride+i];
    }

	// set the pruning vectors
	if (pruningvector == nullptr)
		for (int i = 0; i < N; i++)
			enumobj.pr[i] = 1.0;
	else
		std::copy(&pruningvector[0], &pruningvector[N], &enumobj.pr[0]);

    // pad enumeration tree to MAX_ENUM_N dimension using virtual basis vectors of length above enumeration bound
    for (int i = N; i < MAX_ENUM_N; ++i)
    {
        // ensure these virtual basis vectors are never used
        enumobj.risq[i] = 2.0 * enumobj.risq[0]; // = 2 * ||b0*||^2
		enumobj.pr[i] = 1.0;
    }

	// set the initial norm bound
	enumobj._A = enumeration_radius;


    // perform enumeration
    enumobj.enumerate_recursive();

    // the virtual basis vectors should never be used
    // if sol is non-zero for these positions then there is an internal error
    for (int i = N; i < MAX_ENUM_N; ++i)
        if (enumobj._sol[i] != 0)
        {
            std::cerr << "[enum]: dim=" << N << ": internal error _sol[" << i << "] != 0." << std::endl;
            return 0.0;
        }

    // write enumeration solution to sol
    for (int i = 0; i < N; ++i)
	{
        sol[i] = enumobj._sol[i];
	}

	// return the squared norm of the solution found
	return (enumobj._A < enumeration_radius) ? enumobj._A : 0.0;
}

FT enumeration_lastone(const int N, const FT *R, const int rowstride, const FT *pruningvector, ZZ* sol)
{
    std::fill(sol, sol+N, ZZ(0));

    if (N > MAX_ENUM_N || N <= 0)
        return 0.0;

    lattice_enum_t<MAX_ENUM_N> enumobj;

    // assumption: enumobj.muT is all-zero
    for (int i = 0; i < N-1; ++i)
    {
        FT* muTi = &enumobj.muT[i][0];
        const FT* Ri = R+(i*rowstride);
        FT Rii_inv = FT(1.0) / Ri[i];
        for (int j = i+1; j < N; ++j)
        {
            // muT[i][j] = <bj,bi*> / ||bi*||^2
            muTi[j] = Ri[j] * Rii_inv;
        }
    }

    for (int i = 0; i < N; ++i)
    {
        // risq[i] = ||bi*||^2
        enumobj.risq[i] = R[i*rowstride+i] * R[i*rowstride+i];
    }

	if (pruningvector == nullptr)
		for (int i = 0; i < N; i++)
			enumobj.pr[i] = 1.0;
	else
		std::copy(&pruningvector[0], &pruningvector[N], &enumobj.pr[0]);

	FT enumeration_radius = 0.0;
	for (int i = 0; i < N; i++)
		enumeration_radius += R[i * rowstride + (N-1)] * R[i * rowstride + (N-1)];
	// enumeration_radius = ||b_{N-1}||^2.

    for (int i = N; i < MAX_ENUM_N; ++i)
    {
        // ensure these virtual basis vectors are never used
        enumobj.risq[i] = 2.0 * enumeration_radius;
		enumobj.pr[i] = 1.0;
    }

	// set the initial norm bound
	enumobj._A = enumeration_radius;

    // perform enumeration
    enumobj.enumerate_recursive_lastone(N-1);

    // the virtual basis vectors should never be used
    // if sol is non-zero for these positions then there is an internal error
    for (int i = N; i < MAX_ENUM_N; ++i)
        if (enumobj._sol[i] != 0)
        {
            std::cerr << "[enum]: dim=" << N << ": internal error _sol[" << i << "] != 0." << std::endl;
            return 0.0;
        }

    // write enumeration solution to sol
    for (int i = 0; i < N; ++i)
	{
        sol[i] = enumobj._sol[i];
	}

	// return the squared norm of the solution found
	return (enumobj._A < enumeration_radius) ? enumobj._A : 0.0;
}

#endif // ENUMLIB_WRAPPER_ENUMERATION_CPP
