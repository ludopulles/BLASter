import numpy as np
import argparse
from blaster.stats import rhf, slope, potential


# Adaption of:
# https://github.com/keeganryan/flatter/blob/main/scripts/visualize_profile.py
#
# This script:
# 1) reads a flatter logfile
# 2) extracts the evolution of root hermite factor as time progresses
# 3) outputs this into a csv file in format "TT,rhf,slope", where TT is the elapsed wall time.
class Delta:
    def __init__(self, start, end, data, prev_data=None, is_reset=False):
        self.start = start
        self.end = end
        self.data = data.copy()
        self.prev = prev_data.copy() if prev_data is not None else None
        self.is_reset = is_reset


class ProfileSet:
    ''' Model the sequence of Gram-Schmidt vector lengths '''

    def __init__(self, fname):
        self.reset()
        self.parse_logfile(fname)

    def reset(self):
        self.__n = 0
        self.__cur_data = np.zeros(1)
        self.__deltas = []
        self.__times = []
        self.__pos = 0
        self._follow = True
        self._expect_reset = False
        self._max_time = None

    def parse_logfile(self, fname):
        with open(fname) as f:
            for line in f:
                if "profile" in line:
                    self.parse_profile_line(line)
        self.position = 1

    def parse_profile_line(self, ln):
        ln = ln.rstrip()
        end_i = ln.find(")")
        start_i = ln.find("profile")
        if "," not in ln[start_i + 8:end_i]:
            # This is a reset line
            dim = int(ln[start_i+8:end_i])
            self.reset_profile(dim)
        else:
            bounds = ln[start_i+8:end_i].split(",")
            start = int(bounds[0])
            end = int(bounds[1])

            ts = ln[ln.find("[")+1:ln.find("]")]
            ts = float(ts)

            end_i = ln.find("]")

            val_pairs = [c.split("+") for c in ln[end_i+1:].split(" ") if len(c) > 0]

            if len(val_pairs) != end - start:
                raise ValueError("Unexpected number of values for profile line")

            vals = [float(v[0])+float(v[1]) for v in val_pairs]

            self.append(start, end, vals, ts)

    def reset_profile(self, dim):
        self._expect_reset = True

    def append(self, start, end, values, time):
        # Expand if necessary
        # assert end <= self.__n

        delta = Delta(start, end, values, is_reset=self._expect_reset)
        self._expect_reset = False

        # Is this the first delta? Apply it.
        if len(self.__deltas) == 0:
            self.__n = end
            self.__cur_data = np.zeros(end)
            self.__cur_data[start:end] = values
        self.__deltas.append(delta)
        self.__times.append(time)
        if self._follow:
            to_adv = len(self.__deltas) - self.__pos
            assert to_adv > 0
            self.advance(to_adv)

    def get_data(self):
        return self.__cur_data.copy()

    def get_time(self):
        return self.__times[self.__pos]

    def get_times(self):
        return self.__times[:]

    def _step_forward(self):
        if self.__pos >= len(self.__deltas) - 1:
            self._follow = True
            return

        delta = self.__deltas[self.__pos + 1]
        s = delta.start
        e = delta.end
        if delta.is_reset:
            if delta.prev is None:
                delta.prev = self.__cur_data.copy()
            self.__cur_data = np.array(delta.data)
        else:
            if delta.prev is None:
                delta.prev = self.__cur_data[s:e].copy()
            self.__cur_data[s:e] = delta.data
        self.__pos += 1

    def _step_backward(self):
        self._follow = False
        if self.__pos <= 0:
            return

        delta = self.__deltas[self.__pos]
        assert delta.prev is not None
        s = delta.start
        e = delta.end

        if delta.is_reset:
            self.__cur_data = delta.prev.copy()
        else:
            self.__cur_data[s:e] = delta.prev
        self.__pos -= 1

    def advance(self, steps):
        if steps < 0:
            steps = -steps
            while steps > 0:
                self._step_backward()
                steps -= 1
        else:
            while steps > 0:
                self._step_forward()
                steps -= 1

    def advance_to_time(self, t):
        # Which index is closest to the specified time?
        ts = np.array(self.__times)

        i = np.argmax(ts > t)
        self.position = i + 1

    @property
    def max_time(self):
        if self._max_time is None:
            self._max_time = np.max(self.__times)
        return self._max_time

    @property
    def position(self):
        return self.__pos + 1

    @position.setter
    def position(self, p):
        delta = p - self.position
        self.advance(delta)

    @property
    def count(self):
        return len(self.__deltas)

    def log_profile(self, fname):
        """
        Output the RHF and slope of the basis profile, as a function of elapsed wall time.
        """
        pairs = []
        while self.__pos + 1 < len(self.__deltas):
            prof = self.get_data()
            pairs.append((self.get_time(), rhf(prof), slope(prof), potential(prof)))
            self._step_forward()

        npairs, sparse_pairs = len(pairs), [pairs[0]]
        for i in range(1, npairs):
            if i == npairs - 1 or abs(pairs[i][1] - pairs[i-1][1]) > 1e-6:
                sparse_pairs.append(pairs[i])

        with open(fname, "w") as f:
            print("it,walltime,rhf,slope,potential", file=f)
            it = 1
            for tt, _rhf, _slope, _pot in sparse_pairs:
                print(f"{it:4d},{tt:.6f},{_rhf:.6f},{_slope:.6f},{_pot:.3f}", file=f)
                it += 1


def convert_logfiles(logfile, outfile):
    prof_set = ProfileSet(logfile)
    prof_set.log_profile(outfile)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str, help="Input: flatter logfile")
    parser.add_argument("outfile", type=str, help="Output: RHF as function of elapsed wall time")
    args = parser.parse_args()
    convert_logfiles(args.logfile, args.outfile)


if __name__ == "__main__":
    __main__()
