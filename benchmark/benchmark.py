#!/usr/bin/python3
import os
import subprocess
import sys

from flatter_conversion import convert_logfiles
from blaster.IO import read_qary_lattice
from blaster.stats import get_profile, rhf, slope


# Specify which lattices we want to test:
mqs = [
    (128, 631),  # latticegen q 2 1 10 p
    (256, 829561),  # latticegen q 2 1 20 p
    (512, 968665207),  # latticegen q 2 1 30 p
    (1024, 968665207),  # latticegen q 2 1 30 p
]
seeds = range(10)
cmd_blaster = "../python3 ../src/app.py -q"
temp_lat = "../output/temp.lat"
other_logs = {m: open(f'./logs_other_{m}.csv', mode='w', encoding='utf8') for (m, q) in mqs}


def is_float(x):
    try:
        x = float(x)
    except:
        return False
    return True


def parse_time_usage(time_output):
    times = time_output.strip().split(" ")
    # parts = time_output.split("\n")[1:4]
    # times = [part.split("\t")[1] for part in parts]
    return {'real': times[0], 'user': times[1], 'sys': times[2]}


def run_command(cmd, logfile=None, capture_time=False, flatter_fail=False):
    print(f"Executing \"{cmd}\".", flush=True)
    if capture_time:
        result = subprocess.run(
            f"/usr/bin/time -f \"%e %U %S\" {cmd}",
            text=True, shell=True, capture_output=True
        )
    else:
        result = subprocess.run(cmd, shell=True)
    if not flatter_fail and result.returncode != 0:
        print(result.stderr)
        if logfile:
            os.remove(logfile)
        exit(1)

    if capture_time:
        return parse_time_usage(result.stderr)
    else:
        return None


def gen_lattice(m, q, seed, path):
    n = m//2
    run_command(f"latticegen -randseed {seed} q {m} {n} {q} q > {path}")


def run_blaster(m, q, seed, path):
    logfile = f"../logs/lll_{m}_{q}_{seed}.csv"
    run_command(f"{cmd_blaster} -i {path} -l {logfile}", logfile)


def run_blaster_deeplll(m, q, seed, path, depth):
    logfile = f"../logs/deeplll{depth}_{m}_{q}_{seed}.csv"
    # outfile = path.replace('input/', f'output/d{depth}_')
    # result = run_command(f"{cmd_blaster} -i {path} -o {outfile} -l {logfile} -d{depth}")
    run_command(f"{cmd_blaster} -i {path} -l {logfile} -d{depth}", logfile)


def run_blaster_bkz(m, q, seed, path, beta, bkz_prog=2):
    logfile = f"../logs/progbkz{beta}_{m}_{q}_{seed}.csv"
    run_command(f"{cmd_blaster} -i {path} -l {logfile} -b{beta} -P{bkz_prog} -t1", logfile)


def run_flatter(m, q, seed, path, num_threads, alpha=None):
    flogfile = f"../logs-flatter/{m}_{q}_{seed}.log"
    cmd = f"OMP_NUM_THREADS={num_threads} FLATTER_LOG={flogfile} flatter -q {path}"
    if alpha:
        cmd = f"{cmd} -alpha {alpha}"
    run_command(cmd, flogfile, flatter_fail=True)

    plogfile = f"../logs/flatter_{m}_{q}_{seed}.csv"
    convert_logfiles(flogfile, plogfile)


def run_fplll(m, q, seed, path):
    cmd = f"fplll {path} > {temp_lat}"
    t = run_command(cmd, capture_time=True)
    prof = get_profile(read_qary_lattice(temp_lat))
    data = {
        'seed': seed, 'type': "fpLLL",
        'slope': f"{slope(prof):.6f}", 'rhf': f"{rhf(prof):.5f}"
    }
    print(','.join(str(v) for k, v in (data | t).items()), file=other_logs[m], flush=True)
    other_logs[m].flush()


def run_KEF21(m, q, seed, path, num_threads):
    cmd = f"optlll -p {num_threads} < {path} > {temp_lat}"
    t = run_command(cmd, capture_time=True)
    prof = get_profile(read_qary_lattice(temp_lat))
    data = {
        'seed': seed, 'type': f"KEF21 ({num_threads} threads)",
        'slope': f"{slope(prof):.6f}", 'rhf': f"{rhf(prof):.5f}"
    }
    print(','.join(str(v) for k, v in (data | t).items()), file=other_logs[m], flush=True)


def __main__():
    lattices = [(m, q, seed, f"../input/{m}_{q}_{seed}") for (m, q) in mqs for seed in seeds]

    for f in other_logs.values():
        print("seed,type,slope,rhf,real (s),user (s),sys (s)", file=f, flush=True)

    has_cmd = False
    for i, arg in enumerate(sys.argv[1:]):
        is_cmd = True
        if arg == 'dim':
            assert 2 + i < len(sys.argv), "dim param expected!"
            dim = int(sys.argv[2 + i])
            assert dim in [m for (m, q) in mqs], "Unknown dimension"
            curq = [q for (m, q) in mqs if m == dim]
            assert len(curq) == 1
            curq = curq[0]
            lattices = [(dim, curq, seed, f"../input/{dim}_{curq}_{seed}") for seed in seeds]
        elif arg == 'lattices':
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
        elif arg == 'pbkz':
            assert 2 + i < len(sys.argv), "beta param expected!"
            beta = int(sys.argv[2 + i])
            for lat in lattices:
                run_blaster_bkz(*lat, beta)
        elif arg == 'flatter':
            assert 2 + i < len(sys.argv), "num_threads param expected!"
            num_threads = int(sys.argv[2 + i])
            alpha = None
            if 3 + i < len(sys.argv) and is_float(sys.argv[3 + i]):
                alpha = float(sys.argv[3 + i])
            for lat in lattices:
                run_flatter(*lat, num_threads, alpha)
        elif arg == 'fplll':
            for lat in lattices:
                run_fplll(*lat)
        elif arg == 'KEF21':
            assert 2 + i < len(sys.argv), "num_threads param expected!"
            num_threads = int(sys.argv[2 + i])
            for lat in lattices:
                run_KEF21(*lat, num_threads)
        else:
            is_cmd = False
        has_cmd = has_cmd or is_cmd

    for f in other_logs.values():
        f.close()

    if not has_cmd:
        print(f"Usage: {sys.argv[0]} [dim d|lattices|lll|deeplll `depth`|"
              f"pbkz `beta`|flatter `num_threads`]")


if __name__ == "__main__":
    __main__()
