#!/usr/bin/env python3
import time
import csv
import psutil
import tracemalloc
import numpy as np
from statistics import mean, stdev

from MatrixFunctions import matmul_basic, matmul_parallel

CONFIG = {
    "warmup_iterations": 5,
    "measurement_iterations": 10,
    "forks": 3,
    "matrix_sizes": [500, 1000, 2000, 5000, 10000],
    "dtype": np.float64,
    "parallel_workers": 8
}


def measure_once(func, A, B):
    p = psutil.Process()
    cpu_before = p.cpu_times()

    tracemalloc.start()
    t0 = time.perf_counter()
    func(A, B)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_after = p.cpu_times()

    return {
        "time": t1 - t0,
        "mem_peak": peak,
        "cpu_user": cpu_after.user - cpu_before.user
    }


def run_benchmark(name, func, n):
    print(f"\n=== Benchmark {name} n={n} ===")

    A = np.random.rand(n, n).astype(CONFIG["dtype"])
    B = np.random.rand(n, n).astype(CONFIG["dtype"])

    results = []

    for fork in range(CONFIG["forks"]):
        print(f"  Fork {fork+1}/{CONFIG['forks']}")

        for _ in range(CONFIG["warmup_iterations"]):
            func(A, B)

        fork_results = []
        for i in range(CONFIG["measurement_iterations"]):
            r = measure_once(func, A, B)
            fork_results.append(r)
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
    all_rows = []

    for n in CONFIG["matrix_sizes"]:
        res_basic = run_benchmark("basic", matmul_basic, n)
        sum_basic = summarize(res_basic)

        res_parallel = run_benchmark(
            "parallel",
            lambda A, B: matmul_parallel(A, B, CONFIG["parallel_workers"]),
            n
        )
        sum_parallel = summarize(res_parallel)

        all_rows.append([
            n,
            sum_basic["time_mean"], sum_parallel["time_mean"],
            sum_basic["mem_MB"], sum_parallel["mem_MB"],
            sum_basic["cpu_user"], sum_parallel["cpu_user"]
        ])

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n",
            "time_basic", "time_parallel",
            "mem_basic_MB", "mem_parallel_MB",
            "cpu_basic", "cpu_parallel"
        ])
        writer.writerows(all_rows)

    print("\nResults saved in benchmark_results.csv")


if __name__ == "__main__":
    main()
