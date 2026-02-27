"""
BLASter lattice reduction: LLL with QR decomposition, Seysen's reduction, and
segments, in which lattice reduction is done in parallel.
"""
from functools import partial
from sys import stderr
from time import perf_counter_ns

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter

# Local imports
from blaster_core import \
    set_debug_flag, set_num_cores, block_lll, block_deep_lll, block_bkz, ZZ_right_matmul
from size_reduction import is_lll_reduced, is_weakly_lll_reduced, size_reduce, seysen_reduce
from stats import get_profile, rhf, slope, potential


class TimeProfile:
    """
    Object containing time spent on different parts when running BLASter.
    """

    def __init__(self, use_seysen: bool = False):
        self._strs = [
            "QR-decomp.", "LLL-red.", "BKZ-red.",
            "Seysen-red." if use_seysen else "Size-red.  ", "Matrix-mul."
        ]
        self.num_iterations = 0
        self.times = [0] * 5

    def tick(self, *times):
        self.num_iterations += 1
        self.times = [x + y for x, y in zip(self.times, times)]

    def __str__(self):
        return (
            f"Iterations: {self.num_iterations}\n" +
            "\n".join(f"t_{{{s:11}}}={t/10**9:10.3f}s" for s, t in zip(self._strs, self.times) if t)
        )


def lll_reduce(B, U, U_seysen, lll_size, delta, depth,
               tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's lattice reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, use deep insertions up to depth `depth`.
    """
    n, is_reduced, offset = B.shape[1], False, 0
    red_fn = partial(block_deep_lll, depth) if depth else block_lll

    # Keep running until the basis is LLL reduced.
    while not is_reduced:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call LLL concurrently on small blocks.
        t2 = perf_counter_ns()
        offset = lll_size // 2 if offset == 0 else 0
        red_fn(R, B, U, delta, offset, lll_size)  # LLL or deep-LLL

        if debug:
            for i in range(offset, n, lll_size):
                j = min(n, i + lll_size)
                # Check whether R_[i:j) is really LLL-reduced.
                assert is_lll_reduced(R[i:j, i:j], delta)

        # Step 3: QR-decompose again because LLL "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        # Step 6: Check whether the basis is weakly-LLL reduced.
        t6 = perf_counter_ns()

        is_reduced = is_weakly_lll_reduced(R, delta)
        tprof.tick(t2 - t1 + t4 - t3, t3 - t2, 0, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"DeepLLL-{depth}" if depth else "LLL", None)
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)


def bkz_reduce(B, U, U_seysen, lll_size, delta, depth,
               beta, bkz_tours, bkz_size, tprof, tracers, debug, use_seysen):
    """
    Perform BLASter's BKZ reduction on basis B, and keep track of the transformation in U.
    If `depth` is supplied, BLASter's deep-LLL is called in between calls of the SVP oracle.
    Otherwise BLASter's LLL is run.
    """
    # BKZ parameters:
    n, tours_done, cur_front = B.shape[1], 0, 0

    lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)

    while tours_done < bkz_tours:
        # Step 1: QR-decompose B, and only store the upper-triangular matrix R.
        t1 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')

        # Step 2: Call BKZ concurrently on small blocks!
        t2 = perf_counter_ns()
        # norm_before = abs(R[cur_front, cur_front])
        block_bkz(beta, R, B, U, delta, cur_front % beta, bkz_size)

        # Step 3: QR-decompose again because BKZ "destroys" the QR decomposition.
        # Note: it does not destroy the bxb blocks, but everything above these: yes!
        t3 = perf_counter_ns()
        R = np.linalg.qr(B, mode='r')
        # assert abs(R[cur_front, cur_front]) <= norm_before

        # Step 4: Seysen reduce or size reduce the upper-triangular matrix R.
        t4 = perf_counter_ns()
        with np.errstate(all='raise'):
            (seysen_reduce if use_seysen else size_reduce)(R, U_seysen)

        # Step 5: Update B and U with transformation from Seysen's reduction.
        t5 = perf_counter_ns()
        ZZ_right_matmul(U, U_seysen)
        ZZ_right_matmul(B, U_seysen)

        t6 = perf_counter_ns()

        tprof.tick(t2 - t1 + t4 - t3, 0, t3 - t2, t5 - t4, t6 - t5)

        # After time measurement:
        prof = get_profile(R, True)  # Seysen did not modify the diagonal of R
        note = (f"BKZ-{beta}", (beta, tours_done, bkz_tours, cur_front))
        for tracer in tracers.values():
            tracer(tprof.num_iterations, prof, note)

        # After printing: update the current location of the 'reduction front'
        if cur_front + beta > n:
            # HKZ-reduction was performed at the end, which is the end of a tour.
            cur_front = 0
            tours_done += 1
        else:
            cur_front += (bkz_size - beta + 1)

        # Perform a final LLL reduction at the end
        lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)


def reduce(
        B, lll_size: int = 64, delta: float = 0.99, cores: int = 1, debug: bool = False,
        verbose: bool = False, logfile: str = None, anim: str = None, depth: int = 0,
        use_seysen: bool = False,
        **kwds
):
    """
    :param B: a basis, consisting of *column vectors*.
    :param delta: delta factor for Lagrange reduction,
    :param cores: number of cores to use, and
    :param lll_size: the block-size for LLL, and
    :param debug: whether or not to debug and print more output on time consumption.
    :param kwds: additional arguments (for BKZ reduction).

    :return: tuple (U, B · U, tprof) where:
        U: the transformation matrix such that B · U is LLL reduced,
        B · U: an LLL-reduced basis,
        tprof: TimeProfile object.
    """
    n, tprof = B.shape[1], TimeProfile(use_seysen)
    lll_size = min(max(2, lll_size), n)

    set_num_cores(cores)
    set_debug_flag(1 if debug else 0)

    tracers = {}
    if verbose:
        def trace_print(_, prof, note):
            log_str = '.'
            if note[0].startswith('BKZ'):
                beta, tour, ntours, touridx = note[1]
                log_str = (f"\nBKZ(β:{beta:3d},t:{tour + 1:2d}/{ntours:2d}, o:{touridx:4d}): "
                           f"slope={slope(prof):.6f}, rhf={rhf(prof):.6f}")
            print(log_str, end="", file=stderr, flush=True)
        tracers['v'] = trace_print

    # Set up logfile
    has_logfile = logfile is not None
    if has_logfile:
        tstart = perf_counter_ns()
        logfile = open(logfile, "w", encoding="utf8")
        print('it,walltime,rhf,slope,potential,note', file=logfile, flush=True)

        def trace_logfile(it, prof, note):
            walltime = (perf_counter_ns() - tstart) * 10**-9
            print(f'{it:4d},{walltime:.6f},{rhf(prof):8.6f},{slope(prof):9.6f},'
                  f'{potential(prof):9.3f},{note[0]}', file=logfile)

        tracers['l'] = trace_logfile

    # Set up animation
    has_animation = anim is not None
    if has_animation:
        fig, ax = plt.subplots()
        ax.set(xlim=[0, n])
        artists = []

        def trace_anim(_, prof, __):
            artists.append(ax.plot(range(n), prof, color="blue"))

        tracers['a'] = trace_anim

    B = B.copy()  # Do not modify B in-place, but work with a copy.
    U = np.identity(n, dtype=np.int64)
    U_seysen = np.identity(n, dtype=np.int64)

    beta = kwds.get("beta")
    try:
        if not beta:
            lll_reduce(B, U, U_seysen, lll_size, delta, depth, tprof, tracers, debug, use_seysen)
        elif beta < 40:
            lll_reduce(B, U, U_seysen, lll_size, delta, 4, tprof, tracers, debug, use_seysen)  # depth is 4 
        else:
            # Parse BKZ parameters:
            bkz_tours = kwds.get("bkz_tours") or 1
            bkz_size = kwds.get("bkz_size") or lll_size
            bkz_prog = kwds.get("bkz_prog") or beta

            # Progressive-BKZ: start running BKZ-beta' for some `beta' >= 40`,
            # then increase the blocksize beta' by `bkz_prog` and run BKZ-beta' again,
            # and repeat this until `beta' = beta`.
            betas = range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog)

            # In the literature on BKZ, it is usual to run LLL before calling the SVP oracle in BKZ.
            # However, it is actually better to preprocess the basis with 4-deep-LLL instead of LLL,
            # before calling the SVP oracle.
            for beta_ in betas:
                bkz_reduce(B, U, U_seysen, lll_size, delta, 4, beta_,
                           bkz_tours if beta_ == beta else 1, bkz_size,
                           tprof, tracers, debug, use_seysen)
    except KeyboardInterrupt:
        pass  # When interrupted, give the partially reduced basis.

    # Close logfile
    if has_logfile:
        logfile.close()

    # Save and/or show the animation
    if has_animation:
        # Saving the animation takes a LONG time.
        if verbose:
            print('\nOutputting animation...', file=stderr)
        fig.tight_layout()
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200)
        # Generate 1920x1080 image:
        plt.gcf().set_size_inches(16, 9)
        # plt.show()
        ani.save(anim, dpi=120, writer=PillowWriter(fps=5))

    return U, B, tprof
