#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import time
import psutil
import tracemalloc
from statistics import mean, stdev

from RealDistributed import distributed_matmul

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

CONFIG = {
    "warmup_iterations": 2,
    "measurement_iterations": 6,
    "forks": 1,
    "matrix_sizes": [2000, 5000, 10000],
    "dtype": np.float64
}



def measure_once(A, B):
    p = psutil.Process()
    cpu_before = p.cpu_times()
    tracemalloc.start()
    t0 = time.perf_counter()
    distributed_matmul(A, B)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpu_after = p.cpu_times()
    return {
        "time": t1 - t0,
        "mem_peak": peak,
        "cpu_user": cpu_after.user - cpu_before.user
    }


def run_benchmark(n):
    if rank == 0:
        print(f"\n=== MPI Benchmark n={n}, size={size} processes ===")
        A = np.random.rand(n, n).astype(CONFIG["dtype"])
        B = np.random.rand(n, n).astype(CONFIG["dtype"])
    else:
        A = None
        B = None

    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)

    results = []

    for fork in range(CONFIG["forks"]):
        if rank == 0:
            print(f"  Fork {fork+1}/{CONFIG['forks']}")
        for _ in range(CONFIG["warmup_iterations"]):
            distributed_matmul(A, B)
        fork_results = []
        for i in range(CONFIG["measurement_iterations"]):
            r = measure_once(A, B)
            fork_results.append(r)
            if rank == 0:
                print(f"    Iter {i+1}: {r['time']:.4f} s, mem={r['mem_peak']/1e6:.2f} MB")
        results.append(fork_results)

    return results


def summarize(results):
    flat_times = [r["time"] for fork in results for r in fork]
    flat_mem = [r["mem_peak"] for fork in results for r in fork]
    flat_cpu = [r["cpu_user"] for fork in results for r in fork]
    return {
        "time_mean": mean(flat_times),
        "time_std": stdev(flat_times),
        "mem_MB": mean(flat_mem) / 1e6,
        "cpu_user": mean(flat_cpu)
    }


def main():
    if rank == 0:
        import csv
        all_rows = []

    for n in CONFIG["matrix_sizes"]:
        results = run_benchmark(n)
        if rank == 0:
            summary = summarize(results)
            print(f"\n>>> SUMMARY n={n}")
            print(f"Time mean: {summary['time_mean']:.4f} s")
            print(f"Time std:  {summary['time_std']:.4f} s")
            print(f"Mem mean:  {summary['mem_MB']:.2f} MB")
            print(f"CPU user:  {summary['cpu_user']:.4f} s")
            all_rows.append([
                n,
                summary["time_mean"],
                summary["time_std"],
                summary["mem_MB"],
                summary["cpu_user"]
            ])

    if rank == 0:
        with open("benchmark_mpi_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "time_mean", "time_std", "mem_MB", "cpu_user"])
            writer.writerows(all_rows)
        print("\nMPI results saved in benchmark_mpi_results.csv")


if __name__ == "__main__":
    main()
