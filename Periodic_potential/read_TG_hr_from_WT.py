import numpy as np
import pandas as pd
from io import TextIOWrapper


def get_integer_from_line(line: str):
    from re import search
    if res := search(r"\d+", line):
        match = res.group()
        if match.isnumeric():
            return int(res.group())
    raise ValueError("missing expected value")


def process_header(file: TextIOWrapper):
    file.readline()
    file.readline()
    n_atoms = get_integer_from_line(file.readline())
    # n_r = get_integer_from_line(file.readline())
    file.readline()
    # FIXME This assumes that there is only 1 line of ones. But it can be multiple lines of 1 and perhaps rather than skipping the line entirely, make it an array of ndegen .
    file.readline()
    return n_atoms


def process_header_non_sparse(file: TextIOWrapper):
    file.readline()
    n_atoms = get_integer_from_line(file.readline())
    file.readline()
    file.readline()
    return n_atoms


def read_data(filename, skiprows, sparse):
    with open(filename) as f:
        n_atoms = process_header(
            f) if sparse == True else process_header_non_sparse(f)
        np_data = np.loadtxt(f,skiprows=skiprows)
        pd_data = pd.DataFrame(np_data)
    return n_atoms, np_data, pd_data


def pick_cell_R(data, r):
    return data[(data[0] == r[0]) & (data[1] == r[1]) & (data[2] == r[2])]
