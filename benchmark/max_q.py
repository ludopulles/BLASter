#!/usr/bin/python3
import os
import signal
import subprocess

ERROR_FILE = open("results/max_q_errors.txt", "w", encoding='utf8')
NUM_TRIALS = 10


def gen_lattice(m, lgq, seed, path):
    cmd = f"latticegen -randseed {seed} q {m} {m // 2} {lgq} p > {path}"
    result = subprocess.run(cmd, text=True, shell=True, capture_output=True)
    if result.returncode != 0:
        print("Error encountered during lattice generation.")
        print(result.stderr)
        exit(1)


# https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
def run_command(cmd, timeout, description):
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    cmd = f"/usr/bin/time -f \"%e\" {cmd}"
    with subprocess.Popen(cmd, text=True, shell=True, preexec_fn=os.setsid,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # Send the signal to all the process groups

            # POSIX _communicate already populated the output so
            # far into the TimeoutExpired exception.
            process.wait()
            print(f"Timeout expired for command \"{cmd}\"", file=ERROR_FILE)
            if exc.stderr is not None:
                print(exc.stderr.decode("utf-8"), file=ERROR_FILE, flush=True)
            return None
        except:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
        result = subprocess.CompletedProcess(process.args, retcode, stdout, stderr)
    if result.returncode != 0:
        print(f"{description} finished with error code {result.returncode}.", file=ERROR_FILE)
        print(result.stderr, file=ERROR_FILE, flush=True)
        return None
    return float(result.stderr.strip())


def run_blaster(m, lgq, seed, path):
    timeout = 5 if m < 256 else 15
    desc = f"BLASter(m={m}, lgq={lgq}, seed={seed})"
    return run_command(f"../python3 ../src/app.py -qi {path}", timeout, desc)


def count_successes(m, lgq, path):
    wt = []
    for seed in range(NUM_TRIALS):
        gen_lattice(m, lgq, seed, path)
        result = run_blaster(m, lgq, seed, path)
        if result:
            wt.append(result)

    # Report times
    mint, avgt, maxt = (min(wt), sum(wt)/len(wt), max(wt)) if wt else (0, 0, 0)
    print(f"{m:4d},{lgq:2d},{len(wt):2d},{mint:6.2f},{avgt:6.2f},{maxt:6.2f}", flush=True)

    # Return number of successes
    return len(wt)


def __main__():
    # Print CSV header
    print("m,lgq,num_success,time_min,time_avg,time_max", flush=True)

    path = "../input.lat"
    for m in range(2**4, 2**9 + 1, 2**4):
        # Decrement lgq until at least one run is successful.
        for lgq in range(30, 65):
            if count_successes(m, lgq, path) == 0:
                break


if __name__ == "__main__":
    __main__()
