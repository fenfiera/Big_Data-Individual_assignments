import random
import numpy as np


def matrix_multiplication(n):
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    b = [[random.random() for _ in range(n)] for _ in range(n)]
    c = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]


def matrix_multiplication_optimised(n):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.dot(a, b)