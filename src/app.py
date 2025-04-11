#!/usr/bin/env python3
"""
Script for running BLASter lattice reduction from the command line
"""

import argparse
from multiprocessing import cpu_count
from sys import stderr
from math import log2, ceil

import numpy as np

# Local imports
from lattice_io import read_qary_lattice, write_lattice
from blaster import reduce
from stats import gaussian_heuristic, rhf, slope, get_profile


def __main__():
    parser = argparse.ArgumentParser(
            prog='BLASter',
            description='LLL-reduce a lattice using a fast, modern implementation',
            epilog='Input/output is formatted as is done in fpLLL')

    # Global settings
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
            '--cores', '-j', type=int, default=cpu_count() // 2,
            help='number of cores to be used')

    # I/O arguments
    parser.add_argument('--input', '-i', type=str, help='Input file (default=stdin)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default=stdout)')
    parser.add_argument('--logfile', '-l', type=str, default=None, help='Logging file')
    parser.add_argument(
            '--profile', '-p', action='store_true', dest='debug',
            help='Give information on the profile of the output basis')
    parser.add_argument(
            '--quiet', '-q', action='store_true',
            help='Quiet mode will not output the output basis')
    parser.add_argument(
            '--anim', '-a', type=str,
            help='Output a gif-file animating the basis profile during lattice reduction')

    # LLL parameters
    parser.add_argument(
            '--delta', type=float, default=0.99,
            help='delta factor for Lovasz condition')
    parser.add_argument(
            '--lll_size', '-L', type=int, default=64,
            help='Size of blocks on which to call LLL/deep-LLL locally & in parallel')
    parser.add_argument(
            '--no-seysen', '-s', action='store_false', dest='use_seysen',
            help='Use size reduction if argument is given. Otherwise use Seysen\'s reduction.')

    # Parameters specific to deep-LLL:
    parser.add_argument(
            '--depth', '-d', type=int, default=0,
            help='Maximum allowed depth for "deep insertions" in deep-LLL. 0 if not desired.')

    # Parameters specific to BKZ:
    parser.add_argument(
            '--beta', '-b', type=int,
            help='Blocksize used within BKZ. 0 if not desired.')
    parser.add_argument(
            '--bkz-tours', '-t', type=int, default=8,
            help='Number of BKZ-tours to perform.')
    parser.add_argument(
            '--bkz-size', '-S', type=int, default=64,
            help='Size of blocks on which to call BKZ locally & in parallel.')
    parser.add_argument(
            '--bkz-prog', '-P', type=int,
            help='Progressive blocksize increment for BKZ.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Perform sanity checks
    assert 0.25 < args.delta and args.delta < 1.0, 'Invalid value for delta!'
    assert args.lll_size >= 2, 'LLL block size must be at least 2!'
    assert not args.depth or not args.beta, 'Cannot run combination of deep-LLL and BKZ!'

    # Read the basis from input (file)
    B = read_qary_lattice(args.input)
    n = B.shape[1]  # rank of basis

    if args.verbose:
        # Experimentally, LLL gives a RHF of 1.02190
        # See: https://github.com/malb/lattice-estimator/blob/main/estimator/reduction.py
        # TODO: adjust slope to the prediction for deep-LLL and BKZ.
        log_slope = log2(1.02190)  # Slope of graph of basis profile, log2(||b_i*||^2).
        log_det = sum(get_profile(B))
        norm_b1 = 2.0**(log_slope * (n-1) + log_det / n)

        comparison = ""
        if np.count_nonzero(B[:, 0]) == 1:
            q = sum(B[:, 0])
            cmp = "<" if norm_b1 < q else ">="
            comparison = f'{cmp} {int(q):d} '
        print(f'E[∥b₁∥] ~ {norm_b1:.2f} {comparison}'
              f'(GH: λ₁ ~ {gaussian_heuristic(B):.2f})',
              file=stderr)

    # Multithreading is used in two places:
    # 1) Matrix multiplication (controlled by Eigen)
    # 2) Lattice reduction in `core/blaster.pyx` (done using Cython's `prange`, which uses OPENMP)
    # Notes: using more cores creates more overhead, so use cores wisely!
    # Re 1): Starting around dimension >500, there is a performance gain using multiple threads
    # Re 2): The program cannot use more cores in lattice reduction
    # than the number of blocks, so do not spawn more than this number.
    args.cores = max(1, min(args.cores, ceil(n / args.lll_size), cpu_count() // 2))

    # Run BLASter lattice reduction on basis B
    U, B_red, tprof = reduce(B, **vars(args))

    # Write B_red to the output file
    print_mat = args.output is not None
    if print_mat and args.output == args.input:
        print_mat = input('WARNING: input & output files are same!\nContinue? (y/n) ') == 'y'
    if print_mat:
        write_lattice(args.output, B_red)
    elif not args.quiet:
        print(B_red.astype(np.int64))

    # Print time consumption
    if args.verbose:
        print('\n', str(tprof), sep="", file=stderr)

    # Print basis profile
    if args.debug:
        prof = get_profile(B_red)
        print('\nProfile = [' + ' '.join([f'{x:.2f}' for x in prof]) + ']\n'
              f'RHF = {rhf(prof):.5f}^n, slope = {slope(prof):.6f}, '
              f'∥b_1∥ = {2.0**prof[0]:.1f}', file=stderr)

    # Assert that applying U on the basis B indeed gives the reduced basis B_red.
    assert (B @ U == B_red).all()


if __name__ == '__main__':
    float_formatter = "{:.2f}".format
    np.set_printoptions(linewidth=1000, threshold=2147483647, suppress=True)
    np.set_printoptions(formatter={'float_kind':float_formatter})
    __main__()
