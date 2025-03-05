/*
MIT License

Copyright (c) 2024 Marc Stevens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef ENUMLIB_ENUMERATION_HPP
#define ENUMLIB_ENUMERATION_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <array>

#include "types.hpp"

#define NOCOUNTS 1

template <int N>
struct lattice_enum_t
{
	typedef std::array<FT, N> fltrow_t;
	typedef std::array<ZZ, N> introw_t;

	/* inputs */
	// mu^T corresponds to R (B=Q*R) with multiplicative corrections: muT[i][j] = R[i][j] / R[i][i]
	// mu^T is the transposed mu in fplll (see also: An LLL Algorithm with Quadratic Complexity, Nguyen, Stehle, 2009.)
	FT muT[N][N];
	// risq[i] is ||bi*||^2, or R[i][i]*R[i][i]
	fltrow_t risq;
	// the *relative* pruning bounds for the squared norm within the projected sublattices.
	fltrow_t pr;

	/* internals */
	FT _A; // overall enumeration bound
	fltrow_t _AA; // enumeration pruning bounds
	introw_t _x, _Dx, _D2x;
	fltrow_t _sol; // to pass to fplll
	fltrow_t _c;
	introw_t _r;
	std::array<FT, N + 1> _l;
	std::array<std::uint64_t, N + 1> _counts;

	FT _sigT[N][N];

	lattice_enum_t()
		: muT(), risq(), pr()
	{
	}

	inline void _update_AA()
	{
		// ensure that the pruning bounds are non-increasing from a basis perspective.
		for (int k = 0; k < N; ++k) {
			_AA[k] = _A * pr[k];
		}
	}

	// compile time parameters for enumerate_recur (without ANY runtime overhead)
	// allows specialization for certain specific cases, e.g., i=0
	template<int i> struct i_tag {};

	template<int i>
	inline void enumerate_recur(i_tag<i>)
	{
		if (_r[i] > _r[i - 1])
			_r[i - 1] = _r[i];
		FT ci = _sigT[i][i];
		FT yi = round(ci);
		ZZ xi = (ZZ)(yi);
		yi = ci - yi;
		FT li = _l[i + 1] + (yi * yi * risq[i]);
#ifndef NOCOUNTS
		++_counts[i];
#endif

		if (li > _AA[i])
			return;

		_Dx[i] = _D2x[i] = (((int)(yi >= 0) & 1) << 1) - 1;
		_c[i] = ci;
		_x[i] = xi;
		_l[i] = li;

		for (int j = _r[i - 1]; j > i - 1; --j)
			_sigT[i - 1][j - 1] = _sigT[i - 1][j] - _x[j] * muT[i - 1][j];

		while (true)
		{
			enumerate_recur(i_tag<i - 1>());

			if (_l[i + 1] == 0.0) {
				++_x[i];
			} else {
				_x[i] += _Dx[i];
				_D2x[i] = -_D2x[i];
				_Dx[i] = _D2x[i] - _Dx[i];
			}

			_r[i - 1] = i;
			FT yi2 = _c[i] - _x[i];
			FT li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
			if (li2 > _AA[i])
				return;
			_l[i] = li2;
			_sigT[i - 1][i - 1] = _sigT[i - 1][i] - _x[i] * muT[i - 1][i];
		}
	}

	inline void enumerate_recur(i_tag<0>)
	{
		static constexpr int i = 0;
		FT ci = _sigT[i][i];
		FT yi = round(ci);
		ZZ xi = (ZZ)(yi);
		yi = ci - yi;
		FT li = _l[i + 1] + (yi * yi * risq[i]);
#ifndef NOCOUNTS
		++_counts[i];
#endif

		if (li > _AA[i])
			return;

		_Dx[i] = _D2x[i] = (((int)(yi >= 0) & 1) << 1) - 1;
		_c[i] = ci;
		_x[i] = xi;
		_l[i] = li;

		while (true)
		{
			enumerate_recur(i_tag<i - 1>());

			if (_l[i + 1] == 0.0) {
				++_x[i];
			} else {
				_x[i] += _Dx[i];
				_D2x[i] = -_D2x[i];
				_Dx[i] = _D2x[i] - _Dx[i];
			}

			FT yi2 = _c[i] - _x[i];
			FT li2 = _l[i + 1] + (yi2 * yi2 * risq[i]);
			if (li2 > _AA[i])
				return;
			_l[i] = li2;
		}
	}


	inline void enumerate_recur(i_tag<-1>)
	{
		if (_l[0] > _A || _l[0] == 0.0)
			return;

		for (int j = 0; j < N; ++j)
			_sol[j] = _x[j];

		_A = _l[0];
		_update_AA();
	}

	inline void enumerate_recursive()
	{
		_update_AA();

		std::fill(_l.begin(), _l.end(), 0.0);
		std::fill(_x.begin(), _x.end(), 0);
		std::fill(_Dx.begin(), _Dx.end(), 0);
		std::fill(_D2x.begin(), _D2x.end(), -1);
		std::fill(_c.begin(), _c.end(), 0.0);

		std::fill(_r.begin(), _r.end(), N-1);
		std::fill_n(&_sigT[0][0], N * N, 0.0);

		std::fill(_sol.begin(), _sol.end(), 0);
		std::fill(_counts.begin(), _counts.end(), 0);

		enumerate_recur(i_tag<N-1>());
	}


	/**
	 * Code for enumeration that finds the shortest vector such that the last coefficient equals 1.
	 * Based on `template<int i> inline void enumerate_recur(i_tag<i>)` method.
	 */
	template<int i>
	inline void enumerate_recur_lastone(i_tag<i> _, int k)
	{
#ifndef NOCOUNTS
		++_counts[i];
#endif

		i_tag sub = i_tag<i - 1>();
		if (i == k) {
			// Force this coefficient to equal 1.
			_x[i] = 1;
			_r[i - 1] = i;
			_l[i] = risq[i];
			_sigT[i - 1][i - 1] = -muT[i - 1][i];

			enumerate_recur(sub);
			return;
		} else enumerate_recur_lastone(sub, k);
	}

	inline void enumerate_recur_lastone(i_tag<0> _, int k) {}

	inline void enumerate_recursive_lastone(int j)
	{
		assert(0 < j && j < N);

		_update_AA();

		std::fill(_l.begin(), _l.end(), 0.0);
		std::fill(_x.begin(), _x.end(), 0);
		std::fill(_Dx.begin(), _Dx.end(), 0);
		std::fill(_D2x.begin(), _D2x.end(), -1);
		std::fill(_c.begin(), _c.end(), 0.0);

		std::fill(_r.begin(), _r.end(), N-1);
		std::fill_n(&_sigT[0][0], N * N, FT(0.0));

		std::fill(_sol.begin(), _sol.end(), 0);
		std::fill(_counts.begin(), _counts.end(), 0);

		enumerate_recur_lastone(i_tag<N - 1>(), j);
		assert(_sol[j] == 1);
	}
};

#endif // ENUMLIB_ENUMERATION_HPP
