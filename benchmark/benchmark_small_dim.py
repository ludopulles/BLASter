#!/usr/bin/python3
import subprocess
import sys
from time import time

from blaster import reduce
from blaster.IO import read_qary_lattice
from blaster.stats import get_profile, rhf, slope


def is_prime(x):
    if x <= 2:
        return x == 2
    for y in range(2, x):
        if y*y > x:
            return True
        if x % y == 0:
            return False
    return True
    # return all(x % i for i in range(2, x))


def next_prime(x):
    while not is_prime(x):
        x += 1
    return x


# Specify which lattices we want to test:
mqs = [(m, next_prime(2**(m//8))) for m in range(48, 256, 16)]
seeds = range(10)
lattice_output = "../output/temp.lat"
output_file = None


def parse_time_usage(time_output):
    times = time_output.strip().split(" ")
    # parts = time_output.split("\n")[1:4]
    # times = [part.split("\t")[1] for part in parts]
    return {'real': times[0], 'user': times[1], 'sys': times[2]}


def output_data(data):
    print(','.join(str(v) for k, v in data.items()), file=output_file)
    print(','.join(str(v) for k, v in data.items()))


def run_command(cmd, instance, env=None):
    # print(f"Executing \"time {cmd}\".", flush=True)
    if not env:
        env = ""
    result = subprocess.run(
        f"{env} /usr/bin/time -f \"%e %U %S\" {cmd}", text=True, shell=True, capture_output=True
    )
    if result.returncode != 0:
        print("ERROR WHILE EXECUTING COMMAND!")
        print(result.stderr)
        exit(1)

    prof = get_profile(read_qary_lattice(lattice_output))
    data = instance | {'slope': f"{slope(prof):.6f}", 'rhf': f"{rhf(prof):.5f}"}
    output_data(data | parse_time_usage(result.stderr))


def exec_blaster(inputfile, depth, instance):
    T0 = time()

    # Read lattice
    B = read_qary_lattice(inputfile)

    # Run lattice reduction
    U, B_red, tprof = reduce(B, depth=depth)

    T1 = time()
    prof = get_profile(B_red)
    data = instance | {'slope': f"{slope(prof):.6f}", 'rhf': f"{rhf(prof):.5f}"}
    output_data(data | {'real': (T1 - T0), 'user': '0', 'sys': 0})


def gen_lattice(m, q, seed, path):
    n = m//2
    run_command(f"latticegen -randseed {seed} q {m} {n} {q} q > {path}")


def run_blaster(m, q, seed, path):
    exec_blaster(path, 0, {'m': m, 'q': q, 'seed': seed, 'type': 'LLL'})


def run_blaster_deeplll(m, q, seed, path, depth):
    exec_blaster(path, depth, {'m': m, 'q': q, 'seed': seed, 'type': f'DeepLLL{depth}'})


def run_flatter(m, q, seed, path, num_threads):
    run_command(
        f"~/.local/bin/flatter {path} {lattice_output}",
        {'m': m, 'q': q, 'seed': seed, 'type': f'Flatter ({num_threads} threads)'},
        env=f"OMP_NUM_THREADS={num_threads}")


def run_fplll(m, q, seed, path):
    cmd = f"~/.local/bin/fplll {path} > {lattice_output}"
    run_command(cmd, {'m': m, 'q': q, 'seed': seed, 'type': 'fpLLL'})


def run_KEF21(m, q, seed, path, num_threads):
    cmd = f"optlll -p {num_threads} < {path} > {lattice_output}"
    run_command(cmd, {'m': m, 'q': q, 'seed': seed, 'type': f'KEF21 ({num_threads} threads)'})


def __main__():
    global output_file

    output_file = open("./small-dimension.csv", mode='w', encoding="utf8")
    print("m,q,seed,type,slope,rhf,real (s),user (s),sys (s)", file=output_file)

    lattices = [(m, q, seed, f"../input/{m}_{q}_{seed}") for (m, q) in mqs for seed in seeds]
    commands_executed = 0
    for i, arg in enumerate(sys.argv[1:]):
        if arg == 'lattices':
            for lat in lattices:
                gen_lattice(*lat)
        elif arg == 'lll':
            for lat in lattices:
                run_blaster(*lat)
        elif arg == 'deeplll':
            assert 2 + i < len(sys.argv), "depth param expected!"
            depth = int(sys.argv[2 + i])
            for lat in lattices:
                run_blaster_deeplll(*lat, depth)
        elif arg == 'flatter':
            assert 2 + i < len(sys.argv), "num_threads param expected!"
            num_threads = int(sys.argv[2 + i])
            for lat in lattices:
                run_flatter(*lat, num_threads)
        elif arg == 'fplll':
            for lat in lattices:
                run_fplll(*lat)
        elif arg == 'KEF21':
            assert 2 + i < len(sys.argv), "num_threads param expected!"
            num_threads = int(sys.argv[2 + i])
            for lat in lattices:
                run_KEF21(*lat, num_threads)
        elif arg == 'all':
            assert 3 + i < len(sys.argv), "depth & num_threads param expected!"
            depth = int(sys.argv[2 + i])
            num_threads = int(sys.argv[3 + i])

            for lat in lattices:
                run_blaster(*lat)
                run_blaster_deeplll(*lat, depth)
                run_flatter(*lat, num_threads)
                run_fplll(*lat)
                run_KEF21(*lat, num_threads)
        else:
            commands_executed -= 1
        commands_executed += 1

    if commands_executed == 0:
        print(f"Usage: {sys.argv[0]} "
              f"[lattices|lll|deeplll `depth`|flatter `num_threads`|fplll|KEF21 `num_threads`]")

    output_file.close()


if __name__ == "__main__":
    __main__()
