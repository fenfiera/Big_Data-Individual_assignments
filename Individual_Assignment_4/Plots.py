#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


df_local = pd.read_csv("benchmark_results.csv")
df_mpi = pd.read_csv("benchmark_mpi_results.csv")

df = pd.merge(df_local, df_mpi, on="n", how="outer")
df = df.sort_values("n")

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def set_limits(values):
    vmax = np.nanmax(values)
    return 0, vmax * 1.15

plt.figure(figsize=(10,7))
plt.plot(df["n"], df["time_basic"], marker="o", label="Basic")
plt.plot(df["n"], df["time_parallel"], marker="o", label="Parallel")
plt.plot(df["n"], df["time_mean"], marker="o", label="Distributed (MPI)")
plt.xlabel("Matrix size n")
plt.ylabel("Time (s)")
plt.title("Execution Time vs Matrix Size")
plt.grid(True)
plt.ylim(set_limits([*df["time_basic"],*df["time_parallel"],*df["time_mean"]]))
plt.legend()
plt.savefig(os.path.join(output_dir, "time_plot.png"))

plt.figure(figsize=(10,7))
plt.plot(df["n"], df["mem_basic_MB"], marker="o", label="Basic")
plt.plot(df["n"], df["mem_parallel_MB"], marker="o", label="Parallel")
plt.plot(df["n"], df["mem_MB"], marker="o", label="Distributed (MPI)")
plt.xlabel("Matrix size n")
plt.ylabel("Memory (MB)")
plt.title("Memory Usage vs Matrix Size")
plt.grid(True)
plt.ylim(set_limits([*df["mem_basic_MB"],*df["mem_parallel_MB"],*df["mem_MB"]]))
plt.legend()
plt.savefig(os.path.join(output_dir, "memory_plot.png"))

plt.figure(figsize=(10,7))
plt.plot(df["n"], df["cpu_basic"], marker="o", label="Basic")
plt.plot(df["n"], df["cpu_parallel"], marker="o", label="Parallel")
plt.plot(df["n"], df["cpu_user"], marker="o", label="Distributed (MPI)")
plt.xlabel("Matrix size n")
plt.ylabel("CPU user time (s)")
plt.title("CPU Usage vs Matrix Size")
plt.grid(True)
plt.ylim(set_limits([*df["cpu_basic"],*df["cpu_parallel"],*df["cpu_user"]]))
plt.legend()
plt.savefig(os.path.join(output_dir, "cpu_plot.png"))


print("Generated plots in ./plots/")
