#!/usr/bin/python3
import os
import signal
import subprocess

mqs = [
    (128, 631),  # latticegen q 2 1 10 p
    (256, 829561),  # latticegen q 2 1 20 p
    (512, 968665207),  # latticegen q 2 1 30 p
    (1024, 968665207),  # latticegen q 2 1 30 p
]
seeds = range(10)
cmd_blaster = "../python3 ../src/app.py -q"
error_file = open("results/optimal_segment_size-errors.txt", "a", encoding='utf8')


def parse_time_usage(time_output):
    """
    Returns (real, user, sys).
    """
    times = time_output.strip().split(" ")
    return (float(times[0]), float(times[1]), float(times[2]))


# https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
def time_run_and_timeout(cmd):
    timeout = 300  # 5 minutes

    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    with subprocess.Popen(cmd, text=True, shell=True, preexec_fn=os.setsid,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # Send the signal to all the process groups

            # POSIX _communicate already populated the output so
            # far into the TimeoutExpired exception.
            process.wait()
            raise
        except:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
    return subprocess.CompletedProcess(process.args, retcode, stdout, stderr)


def run_command(cmd):
    cmd = f"/usr/bin/time -f \"%e %U %S\" {cmd}"
    try:
        result = time_run_and_timeout(cmd)
    except subprocess.TimeoutExpired as exc:
        print(f"Timeout expired for command \"{cmd}\"", file=error_file)
        if exc.stderr is not None:
            print(exc.stderr.decode("utf-8"), file=error_file, flush=True)
        return None
    if result.returncode != 0:
        print(f"Non zero return code encountered for command \"{cmd}\"", file=error_file)
        print(result.stderr, file=error_file, flush=True)
        return None

    return parse_time_usage(result.stderr)


def run_blaster(m, q, seed, LLLsize):
    logfile = f"../logs/optimal_segment_size/lll_{m}_{q}_{seed}_{LLLsize}.csv"
    path = f"../input/{m}_{q}_{seed}"
    return run_command(f"{cmd_blaster} -i {path} -l {logfile} -L {LLLsize}")


def run_blaster_deeplll(m, q, seed, LLLsize, depth):
    logfile = f"../logs/optimal_segment_size/deeplll{depth}_{m}_{q}_{seed}_{LLLsize}.csv"
    path = f"../input/{m}_{q}_{seed}"
    return run_command(f"{cmd_blaster} -i {path} -l {logfile} -L {LLLsize} -d{depth}")


def __main__():
    # Print CSV header
    print("m,q,LLLsize,seed,real,user,sys", flush=True)
    for (m, q) in mqs:
        for LLLsize in range(32, 130, 2):
            if (
                m < 512
                or (m == 512 and LLLsize not in [92, 94, 98, 104])
                or (m == 1024 and LLLsize not in [76, 86, 90, 104, 106, 108, 110])
            ):
                continue
            if LLLsize > m:
                break
            for seed in range(10):
                res = run_blaster(m, q, seed, LLLsize)
                if res:
                    print(f"{m:4d},{q:9d},{LLLsize},{seed:1d},"
                          f"{res[0]:6.2f},{res[1]:6.2f},{res[2]:6.2f}", flush=True)


if __name__ == "__main__":
    __main__()
