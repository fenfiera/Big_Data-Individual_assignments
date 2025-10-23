import pytest
from matrix import *

@pytest.mark.parametrize("n", [10, 128, 512])
def test_matrix_multiplication(benchmark, n):
    result = benchmark.pedantic(matrix_multiplication, args=(n,), iterations=10, rounds=3)

@pytest.mark.parametrize("n", [10, 128, 512, 1024])
def test_matrix_multiplication_optimised(benchmark, n):
    result = benchmark.pedantic(matrix_multiplication_optimised, args=(n,), iterations=10, rounds=3)