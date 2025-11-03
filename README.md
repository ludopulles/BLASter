# BLASter

BLASter is a proof of concept of an LLL-like lattice reduction algorithm that uses:

- parallelization,
- segmentation,
- Seysen's reduction instead of size reduction, and
- a linear algebra library.

## Disclaimer

The goal of this software is to showcase speed ups that are possible in lattice reduction software.
This software is a *proof of concept*!

In particular, we **do not**:

- guarantee the algorithm terminates, nor claim its output is correct on all lattices,
- support lattices with large entries,
- consider issues / PRs that improve efficiency or robustness,
- actively maintain this software.

We **do**:

- happily answer any questions to explain design choices phrased as: *"Why is X done in Y way?"*. The answer may, in many cases, be: "because it is faster in practice".
- encourage the cryptographic community to build a new robust lattice reduction library incorporating the ideas in this proof of concept.

## Requirements

- python3
- Cython version 3.0 or later
- Python modules: `cysignals numpy setuptools matplotlib` (installed system-wide or locally through `make venv`)
- The [Eigen library](https://libeigen.gitlab.io/) version 3 or later (installed system-wide or locally through `make eigen3`)

Optional:

- Python module: virtualenv or venv (for creating a local virtual environment to install python3 modules).
- fplll (for generating q-ary lattices with the `latticegen` command)

## Building

1. (optional) Run `make eigen3` to install the Eigen (version 3.4.0) in a subdirectory.
2. (optional) Run `make venv` to create a local virtual environment and install the required python3 modules.
3. Run `make` to compile all the Cython files in `core/`.

## Debugging

- Debug the C++/Cython code with the `libasan` and `libubsan` sanitizers by running `make cython-gdb`.
    These sanitizers check for memory leaks, out of bounds accesses, and undefined behaviour.
- When executing the script `src/app.py`, preload libasan as follows:
    `LD_PRELOAD=$(gcc -print-file-name=libasan.so) ./python3 src/app.py -pvi INPUTFILE`
- If you want to run the program with the `gdb` debugger, read the [Cython documentation](https://cython.readthedocs.io/en/stable/src/userguide/debugging.html#running-the-debugger), for more info.

## Running

*Note: you first need to build the software, see [above](#building).*

You can run the software from the command line by executing the script `src/app.py`.
For example, `./python3 src/app.py -pvi INPUTFILE` LLL-reduces a lattice in file `INPUTFILE` and outputs it to standard output, and provides additional information to standard error.

To use the software from within your own Python code, call the function `reduce` in the file `src/blaster.py`.

### Input file format
The lattice input format is the same as what is supported by [NTL](https://github.com/libntl/ntl), [FPLLL](https://github.com/fplll/fplll) and [flatter](https://github.com/keeganryan/flatter).
That is, to specify a rank-`k` lattice in `n`-dimensional Euclidean space, the file or input should be of the form:

```
[[a_11 a_12 ... a_1n]
[a_21 a_22 ... a_2n]
...
[a_k1 a_k2 ... a_kn]]
```

Notes:

- the final closing `]` may be put on a new line,
- the input parser is insensitive to extra whitespace in almost all cases.

## Examples

Run `./python3 src/app.py -h` to see all available command line arguments.

### LLL
To generate one *BLASter* data point in [Figure 3](https://eprint.iacr.org/2025/774.pdf), run the following command, which should give (up to timing differences) the following output.

```ShellSession
$ time latticegen -randseed 0 q 128 64 631 q | ./python3 src/app.py -pqv
E[∥b₁∥] ~ 393.44 < 631 (GH: λ₁ ~ 68.77)
........
Iterations: 8
t_{QR-decomp. }=     0.006s
t_{LLL-red.   }=     0.041s
t_{Seysen-red.}=     0.013s
t_{Matrix-mul.}=     0.007s
Profile = [8.56 8.66 8.54 8.40 8.32 8.41 8.35 8.23 8.08 8.11 8.05 8.01 7.88 7.84 7.86 7.74 7.72 7.60 7.59 7.70 7.53 7.49 7.40 7.28 7.08 7.10 7.04 7.03 7.07 6.94 6.96 6.79 6.76 6.74 6.60 6.44 6.31 6.23 6.15 6.26 6.22 6.20 6.08 6.06 6.16 6.01 5.83 5.79 5.69 5.55 5.45 5.34 5.36 5.27 5.16 5.37 5.28 5.09 5.01 5.00 4.95 4.76 4.83 4.70 4.57 4.60 4.39 4.34 4.16 4.08 4.12 3.99 4.01 3.91 3.97 3.82 3.75 3.66 3.57 3.58 3.47 3.44 3.45 3.38 3.26 3.20 3.15 3.20 3.07 3.05 2.89 2.87 2.86 2.89 2.74 2.72 2.52 2.39 2.35 2.28 2.16 2.09 1.97 1.87 2.02 2.01 1.97 1.92 1.78 1.60 1.59 1.60 1.57 1.54 1.61 1.46 1.43 1.47 1.34 1.18 1.13 1.08 1.02 0.87 0.80 0.87 0.84 0.79]
RHF = 1.02142^n, slope = -0.063828, ∥b_1∥ = 378.7

real	0m0.747s
user	0m0.501s
sys	0m0.042s
```

The argument `-p` outputs the basis profile, i.e., the binary logarithm (log\_2) of the norms of the Gram--Schmidt vectors.
The argument `-q` suppresses outputting the reduced basis to standard output.
The argument `-v` gives extra information regarding the reduced basis, i.e., root Hermite factor is 1.02142, the slope equals -0.063828, and the first basis vector has norm 378.7.

### DeepLLL
To generate one *BLASterDeepLLL-4* data point in [Figure 6](https://eprint.iacr.org/2025/774.pdf), run:

```ShellSession
$ time latticegen -randseed 0 q 1024 512 968665207 q | ./python3 src/app.py -d4 -pqv
E[∥b₁∥] ~ 131183490632416.14 >= 968665207 (GH: λ₁ ~ 240990.35)
.....................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
Iterations: 1045
t_{QR-decomp. }=    94.155s
t_{LLL-red.   }=    24.336s
t_{Seysen-red.}=    69.988s
t_{Matrix-mul.}=   156.098s
Profile = [29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.85 29.76 29.78 29.71 29.69 29.69 29.67 29.63 29.57 29.57 29.65 29.55 29.46 29.53 29.38 29.37 29.26 29.22 29.23 29.14 29.14 29.10 28.93 29.01 28.93 28.80 28.83 28.82 28.74 28.67 28.72 28.62 28.55 28.53 28.38 28.41 28.41 28.27 28.30 28.34 28.33 28.23 28.20 28.19 28.14 28.15 28.11 27.96 27.83 27.86 27.88 27.79 27.75 27.69 27.72 27.59 27.58 27.49 27.59 27.49 27.39 27.36 27.28 27.21 27.18 27.12 27.09 27.04 26.91 27.06 26.92 26.98 26.86 26.88 26.87 26.80 26.79 26.78 26.73 26.71 26.58 26.58 26.49 26.48 26.38 26.37 26.33 26.29 26.25 26.24 26.16 26.06 26.07 26.00 26.00 25.86 25.89 25.89 25.78 25.75 25.68 25.62 25.55 25.63 25.59 25.58 25.43 25.48 25.51 25.38 25.26 25.20 25.24 25.26 25.10 25.17 25.05 25.02 24.98 24.95 24.93 24.83 24.82 24.68 24.73 24.58 24.69 24.58 24.54 24.46 24.45 24.38 24.39 24.35 24.27 24.33 24.19 24.25 24.13 24.11 23.97 24.06 24.01 23.96 23.92 23.86 23.84 23.79 23.68 23.73 23.62 23.57 23.55 23.41 23.36 23.37 23.31 23.30 23.31 23.20 23.12 23.09 23.07 22.97 22.90 22.95 22.84 22.92 22.82 22.82 22.80 22.77 22.63 22.57 22.64 22.55 22.61 22.50 22.51 22.39 22.42 22.26 22.27 22.24 22.17 22.10 22.10 22.03 22.06 22.02 21.87 21.79 21.82 21.71 21.68 21.65 21.60 21.57 21.58 21.50 21.52 21.43 21.51 21.41 21.38 21.31 21.33 21.25 21.19 21.09 21.03 20.93 20.97 20.88 20.83 20.86 20.73 20.72 20.71 20.60 20.61 20.62 20.56 20.45 20.50 20.39 20.38 20.26 20.26 20.29 20.19 20.18 20.15 20.16 19.98 20.00 19.90 19.86 19.90 19.84 19.82 19.86 19.71 19.74 19.66 19.58 19.55 19.43 19.44 19.44 19.41 19.26 19.22 19.27 19.29 19.22 19.10 19.09 19.07 18.95 18.87 18.87 18.85 18.90 18.84 18.71 18.65 18.70 18.60 18.62 18.53 18.49 18.46 18.45 18.42 18.30 18.34 18.32 18.15 18.17 18.15 18.05 18.03 17.93 17.89 17.84 17.83 17.83 17.85 17.66 17.80 17.64 17.61 17.52 17.53 17.52 17.41 17.46 17.44 17.35 17.30 17.29 17.20 17.18 17.11 17.01 16.98 16.98 16.98 16.98 16.93 16.84 16.84 16.73 16.57 16.58 16.53 16.41 16.39 16.39 16.36 16.34 16.36 16.20 16.10 16.18 16.12 16.11 16.09 16.00 15.98 16.02 15.95 15.95 15.89 15.83 15.80 15.77 15.63 15.70 15.62 15.50 15.49 15.47 15.35 15.29 15.28 15.14 15.15 15.05 15.00 14.95 14.94 14.89 14.92 14.81 14.81 14.71 14.73 14.63 14.70 14.65 14.67 14.58 14.48 14.52 14.53 14.40 14.34 14.24 14.31 14.30 14.28 14.20 14.14 14.14 14.06 13.94 13.83 13.89 13.77 13.71 13.67 13.71 13.64 13.52 13.53 13.47 13.45 13.44 13.38 13.37 13.36 13.25 13.28 13.25 13.10 13.02 13.03 13.03 12.95 12.94 12.93 12.89 12.84 12.73 12.78 12.66 12.72 12.66 12.67 12.52 12.55 12.45 12.31 12.36 12.30 12.30 12.11 12.14 12.13 12.03 12.01 12.03 12.01 11.95 11.95 11.91 11.74 11.69 11.82 11.72 11.71 11.54 11.58 11.51 11.53 11.46 11.39 11.34 11.30 11.28 11.17 11.15 11.11 11.09 11.00 10.94 10.93 10.87 10.79 10.76 10.75 10.70 10.62 10.63 10.60 10.61 10.55 10.56 10.56 10.39 10.31 10.39 10.34 10.22 10.19 10.10 10.15 10.05 9.91 10.03 9.90 9.89 9.83 9.74 9.71 9.73 9.64 9.58 9.53 9.47 9.51 9.46 9.37 9.35 9.28 9.29 9.27 9.11 9.17 9.16 9.08 9.15 8.96 8.92 8.81 8.88 8.71 8.79 8.78 8.75 8.69 8.51 8.57 8.48 8.46 8.50 8.46 8.42 8.27 8.20 8.17 8.14 8.03 8.08 8.04 7.93 8.01 7.98 7.97 7.92 7.81 7.84 7.75 7.66 7.73 7.56 7.51 7.55 7.41 7.40 7.36 7.27 7.31 7.31 7.33 7.19 7.16 7.06 7.06 6.98 7.05 6.90 6.91 6.83 6.84 6.73 6.73 6.69 6.61 6.57 6.58 6.61 6.49 6.41 6.36 6.28 6.25 6.28 6.28 6.11 6.09 6.06 5.99 5.93 5.81 5.80 5.85 5.87 5.89 5.80 5.70 5.66 5.63 5.61 5.55 5.50 5.50 5.39 5.34 5.29 5.13 5.24 5.20 5.04 5.21 5.05 5.04 5.02 4.95 4.94 4.86 4.78 4.80 4.75 4.67 4.57 4.61 4.56 4.42 4.47 4.43 4.37 4.33 4.24 4.28 4.16 4.20 4.09 4.05 4.04 4.00 4.00 3.93 3.82 3.88 3.87 3.82 3.66 3.55 3.64 3.67 3.52 3.47 3.39 3.45 3.36 3.38 3.25 3.18 3.17 3.16 3.05 2.93 3.08 2.98 2.97 3.01 2.90 2.86 2.67 2.75 2.73 2.67 2.62 2.55 2.57 2.52 2.44 2.40 2.24 2.33 2.33 2.24 2.21 2.13 2.07 2.01 2.01 1.97 1.87 1.97 1.85 1.79 1.80 1.69 1.67 1.61 1.68 1.51 1.56 1.50 1.39 1.35 1.36 1.35 1.32 1.28 1.23 1.25 1.13 1.00 1.00 0.97 0.99 0.93 0.84 0.85 0.84 0.73 0.70 0.59 0.64 0.57 0.59 0.53 0.42 0.43 0.40 0.39 0.36 0.23 0.17 0.13 0.19 0.14 0.05 0.02 -0.00 -0.07 -0.14 -0.16 -0.21 -0.17 -0.27 -0.30 -0.36 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 -0.00 -0.00 0.00 0.00 -0.00 -0.00 -0.00 -0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00]
RHF = 1.01015^n, slope = -0.036847, ∥b_1∥ = 968665207.0

real	5m47.925s
user	18m12.336s
sys	0m23.005s
```

### (Progressive) BKZ
To generate one *BLASterBKZ-60* data point in [Figure 3](https://eprint.iacr.org/2025/774.pdf), run the command found below.
This command runs progressive BKZ (with 4-deep-LLL before SVP calls) with increasing block sizes `40, 42, ..., 60` performing one tour per block size.
Moreover, `-l` will record intermediate progress in the file `logfile.csv`.

```ShellSession
$ time latticegen -randseed 0 q 128 64 631 q | ./python3 src/app.py -b60 -t1 -P2 -l logfile.csv -pqv
E[∥b₁∥] ~ 393.44 < 631 (GH: λ₁ ~ 68.77)
........................
BKZ(β: 40,t: 1/ 1, o:   0): slope=-0.043332, rhf=1.014315.......
BKZ(β: 40,t: 1/ 1, o:  25): slope=-0.042374, rhf=1.014315.....
BKZ(β: 40,t: 1/ 1, o:  50): slope=-0.041580, rhf=1.014195....
BKZ(β: 40,t: 1/ 1, o:  75): slope=-0.041099, rhf=1.013987.
BKZ(β: 40,t: 1/ 1, o: 100): slope=-0.040978, rhf=1.013987...
BKZ(β: 42,t: 1/ 1, o:   0): slope=-0.040810, rhf=1.013356..
BKZ(β: 42,t: 1/ 1, o:  23): slope=-0.040542, rhf=1.013356...
BKZ(β: 42,t: 1/ 1, o:  46): slope=-0.040666, rhf=1.013356..
BKZ(β: 42,t: 1/ 1, o:  69): slope=-0.040395, rhf=1.013356..
BKZ(β: 42,t: 1/ 1, o:  92): slope=-0.040275, rhf=1.012497...
BKZ(β: 44,t: 1/ 1, o:   0): slope=-0.040120, rhf=1.012497....
BKZ(β: 44,t: 1/ 1, o:  21): slope=-0.039930, rhf=1.012497.
BKZ(β: 44,t: 1/ 1, o:  42): slope=-0.039651, rhf=1.012497...
BKZ(β: 44,t: 1/ 1, o:  63): slope=-0.039616, rhf=1.012497..
BKZ(β: 44,t: 1/ 1, o:  84): slope=-0.039546, rhf=1.012497.
BKZ(β: 44,t: 1/ 1, o: 105): slope=-0.039613, rhf=1.012497...
BKZ(β: 46,t: 1/ 1, o:   0): slope=-0.039689, rhf=1.012497....
BKZ(β: 46,t: 1/ 1, o:  19): slope=-0.039445, rhf=1.012497...
BKZ(β: 46,t: 1/ 1, o:  38): slope=-0.039244, rhf=1.012497..
BKZ(β: 46,t: 1/ 1, o:  57): slope=-0.039485, rhf=1.012497....
BKZ(β: 46,t: 1/ 1, o:  76): slope=-0.039283, rhf=1.012497...
BKZ(β: 46,t: 1/ 1, o:  95): slope=-0.039427, rhf=1.012497....
BKZ(β: 48,t: 1/ 1, o:   0): slope=-0.039296, rhf=1.012497.
BKZ(β: 48,t: 1/ 1, o:  17): slope=-0.038946, rhf=1.012497..
BKZ(β: 48,t: 1/ 1, o:  34): slope=-0.038779, rhf=1.012497...
BKZ(β: 48,t: 1/ 1, o:  51): slope=-0.038998, rhf=1.012497...
BKZ(β: 48,t: 1/ 1, o:  68): slope=-0.038829, rhf=1.012497..
BKZ(β: 48,t: 1/ 1, o:  85): slope=-0.038537, rhf=1.012497..
BKZ(β: 50,t: 1/ 1, o:   0): slope=-0.038765, rhf=1.011904..
BKZ(β: 50,t: 1/ 1, o:  15): slope=-0.038361, rhf=1.011904..
BKZ(β: 50,t: 1/ 1, o:  30): slope=-0.038127, rhf=1.011904...
BKZ(β: 50,t: 1/ 1, o:  45): slope=-0.038089, rhf=1.011904.
BKZ(β: 50,t: 1/ 1, o:  60): slope=-0.038368, rhf=1.011904..
BKZ(β: 50,t: 1/ 1, o:  75): slope=-0.038186, rhf=1.011904...
BKZ(β: 50,t: 1/ 1, o:  90): slope=-0.038035, rhf=1.011904..
BKZ(β: 52,t: 1/ 1, o:   0): slope=-0.038113, rhf=1.011904..
BKZ(β: 52,t: 1/ 1, o:  13): slope=-0.038140, rhf=1.011904....
BKZ(β: 52,t: 1/ 1, o:  26): slope=-0.038001, rhf=1.011904.
BKZ(β: 52,t: 1/ 1, o:  39): slope=-0.037897, rhf=1.011904.....
BKZ(β: 52,t: 1/ 1, o:  52): slope=-0.037875, rhf=1.011904..
BKZ(β: 52,t: 1/ 1, o:  65): slope=-0.037652, rhf=1.011904..
BKZ(β: 52,t: 1/ 1, o:  78): slope=-0.037847, rhf=1.011904...
BKZ(β: 54,t: 1/ 1, o:   0): slope=-0.038111, rhf=1.011904.
BKZ(β: 54,t: 1/ 1, o:  11): slope=-0.037940, rhf=1.011904...
BKZ(β: 54,t: 1/ 1, o:  22): slope=-0.037949, rhf=1.011904..
BKZ(β: 54,t: 1/ 1, o:  33): slope=-0.037908, rhf=1.011904.
BKZ(β: 54,t: 1/ 1, o:  44): slope=-0.037818, rhf=1.011904.
BKZ(β: 54,t: 1/ 1, o:  55): slope=-0.037928, rhf=1.011904..
BKZ(β: 54,t: 1/ 1, o:  66): slope=-0.038078, rhf=1.011904..
BKZ(β: 54,t: 1/ 1, o:  77): slope=-0.038205, rhf=1.011904..
BKZ(β: 56,t: 1/ 1, o:   0): slope=-0.038343, rhf=1.011904.
BKZ(β: 56,t: 1/ 1, o:   9): slope=-0.038148, rhf=1.011904.
BKZ(β: 56,t: 1/ 1, o:  18): slope=-0.038094, rhf=1.011904..
BKZ(β: 56,t: 1/ 1, o:  27): slope=-0.037886, rhf=1.011904..
BKZ(β: 56,t: 1/ 1, o:  36): slope=-0.037807, rhf=1.011904...
BKZ(β: 56,t: 1/ 1, o:  45): slope=-0.037415, rhf=1.011904..
BKZ(β: 56,t: 1/ 1, o:  54): slope=-0.037376, rhf=1.011904.
BKZ(β: 56,t: 1/ 1, o:  63): slope=-0.037486, rhf=1.011904..
BKZ(β: 56,t: 1/ 1, o:  72): slope=-0.037483, rhf=1.011904...
BKZ(β: 56,t: 1/ 1, o:  81): slope=-0.037495, rhf=1.011904....
BKZ(β: 58,t: 1/ 1, o:   0): slope=-0.037716, rhf=1.011904.
BKZ(β: 58,t: 1/ 1, o:   7): slope=-0.037442, rhf=1.011904.
BKZ(β: 58,t: 1/ 1, o:  14): slope=-0.037704, rhf=1.011904..
BKZ(β: 58,t: 1/ 1, o:  21): slope=-0.037900, rhf=1.011904....
BKZ(β: 58,t: 1/ 1, o:  28): slope=-0.037714, rhf=1.011904...
BKZ(β: 58,t: 1/ 1, o:  35): slope=-0.037551, rhf=1.011904..
BKZ(β: 58,t: 1/ 1, o:  42): slope=-0.037445, rhf=1.011904.
BKZ(β: 58,t: 1/ 1, o:  49): slope=-0.037212, rhf=1.011904.
BKZ(β: 58,t: 1/ 1, o:  56): slope=-0.037023, rhf=1.011904.
BKZ(β: 58,t: 1/ 1, o:  63): slope=-0.037201, rhf=1.011904..
BKZ(β: 58,t: 1/ 1, o:  70): slope=-0.037115, rhf=1.011904..
BKZ(β: 58,t: 1/ 1, o:  77): slope=-0.037202, rhf=1.011904....
BKZ(β: 60,t: 1/ 1, o:   0): slope=-0.037201, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:   5): slope=-0.037337, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  10): slope=-0.037405, rhf=1.011904..
BKZ(β: 60,t: 1/ 1, o:  15): slope=-0.037518, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  20): slope=-0.037440, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  25): slope=-0.037447, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  30): slope=-0.037260, rhf=1.011904...
BKZ(β: 60,t: 1/ 1, o:  35): slope=-0.037266, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  40): slope=-0.037381, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  45): slope=-0.036954, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  50): slope=-0.036800, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  55): slope=-0.036753, rhf=1.011904.
BKZ(β: 60,t: 1/ 1, o:  60): slope=-0.036773, rhf=1.011904..
BKZ(β: 60,t: 1/ 1, o:  65): slope=-0.036379, rhf=1.011904..
BKZ(β: 60,t: 1/ 1, o:  70): slope=-0.036526, rhf=1.011904..
Iterations: 304
t_{QR-decomp. }=     0.212s
t_{LLL-red.   }=     0.297s
t_{BKZ-red.   }=    12.741s
t_{Seysen-red.}=     0.450s
t_{Matrix-mul.}=     0.282s
Profile = [6.84 6.80 6.79 6.74 6.74 6.73 6.63 6.67 6.56 6.54 6.53 6.53 6.45 6.44 6.38 6.49 6.58 6.57 6.43 6.50 6.43 6.40 6.28 6.29 6.18 6.17 6.01 6.00 5.99 5.93 5.98 5.93 5.84 5.83 5.86 5.81 5.70 5.69 5.59 5.62 5.59 5.47 5.42 5.34 5.35 5.31 5.21 5.18 5.26 5.17 5.04 4.99 5.00 5.02 4.97 4.86 4.86 4.78 4.70 4.76 4.65 4.65 4.58 4.53 4.53 4.42 4.31 4.42 4.30 4.29 4.18 4.33 4.34 4.28 4.29 4.24 4.23 4.22 4.17 4.15 4.12 4.10 4.04 3.99 3.98 3.94 3.87 3.87 3.86 3.83 3.81 3.74 3.71 3.62 3.62 3.49 3.51 3.42 3.51 3.40 3.32 3.34 3.37 3.31 3.23 3.24 3.15 3.12 3.08 3.04 2.89 2.88 2.90 2.80 2.75 2.76 2.79 2.64 2.68 2.65 2.54 2.46 2.41 2.45 2.39 2.38 2.25 2.18]
RHF = 1.01190^n, slope = -0.036484, ∥b_1∥ = 114.2

real	0m14.563s
user	0m17.605s
sys	0m0.045s
```
