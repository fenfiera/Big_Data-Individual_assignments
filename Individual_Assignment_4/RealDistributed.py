#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def distributed_matmul(A, B):
    n = A.shape[0]
    rows_per_rank = n // size
    extra = n % size

    if rank < extra:
        start = rank * (rows_per_rank + 1)
        end = start + rows_per_rank + 1
    else:
        start = rank * rows_per_rank + extra
        end = start + rows_per_rank

    local_A = A[start:end, :]
    local_C = local_A @ B

    counts = []
    displs = []
    offset = 0
    for r in range(size):
        if r < extra:
            c = (rows_per_rank + 1) * n
        else:
            c = rows_per_rank * n
        counts.append(c)
        displs.append(offset)
        offset += c

    if rank == 0:
        C = np.empty((n, n), dtype=A.dtype)
    else:
        C = None

    comm.Gatherv(
        sendbuf=local_C.flatten(),
        recvbuf=(C.flatten() if rank == 0 else None, counts, displs, MPI.DOUBLE),
        root=0
    )

    return C
