"""
Read/write a matrix in fplll format, to/from numpy array.
"""
import numpy as np


def read_qary_lattice(input_file=None):
    """
    Read a matrix from a file, or from stdin.
    :param input_file: file name, or when None, read from stdin.
    :return: a matrix consisting of column vectors.
    """
    data = []
    if input_file is None:
        data.append(input().strip())
        while data[-1] != ']' and data[-1][-2] != ']':
            data.append(input().strip())
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data.append(f.readline().strip())
            while data[-1] != ']' and data[-1][-2] != ']':
                data.append(f.readline().strip())

    # Strip away starting '[' and ending ']'
    assert data[0][0] == '[' and data[-1][-1] == ']'
    data[0] = data[0][1:]
    if data[-1] == ']':
        # Flatter and fpLLL output ']' on a separate line instead of '[<data of last row>]]'
        data.pop()
    else:
        data[-1] = data[-1][:-1]

    # Convert data to list of integers
    data = [list(map(int, line[1:-1].strip().split(' '))) for line in data]

    if np.count_nonzero(data[-1]) == 1:
        # The q-ary vectors are at the back, so reverse the basis vectors in place.
        data.reverse()

    # Use column vectors.
    return np.ascontiguousarray(np.array(data, dtype=np.int64).transpose())


def write_lattice(basis, output_file=None):
    """
    Outputs a basis with column vectors to a file in fplll format.
    :param output_file: file name
    :param basis: the matrix to output
    """
    basis = basis.transpose()

    if output_file is None:
        print('[', end='')
        for (i, v) in enumerate(basis):
            print('[' + ' '.join(map(str, v)), end=']\n' if i < len(basis) - 1 else ']]\n')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[')
            for (i, v) in enumerate(basis):
                f.write('[' + ' '.join(map(str, v)) + (']\n' if i < len(basis) - 1 else ']]\n'))
