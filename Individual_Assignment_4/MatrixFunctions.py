#!/usr/bin/env python3
import numpy as np
from multiprocessing import Pool, cpu_count


def matmul_basic(A, B):
    return A @ B


def _row_mult(args):
    row, B = args
    return row @ B

def matmul_parallel(A, B, workers=None):
    if workers is None:
        workers = cpu_count()
    with Pool(processes=workers) as p:
        rows = [(A[i, :], B) for i in range(A.shape[0])]
        result_rows = p.map(_row_mult, rows)
    return np.vstack(result_rows)
