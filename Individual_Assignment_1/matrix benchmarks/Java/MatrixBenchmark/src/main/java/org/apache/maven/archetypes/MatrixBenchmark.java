package org.apache.maven.archetypes;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
public class MatrixBenchmark {

	@Param({"10", "128", "512", "1024"})
	private int n;

	private double[][] a;
	private double[][] b;
	private double[][] c;
	private Random random = new Random();

	@Setup(Level.Invocation)
	public void setUp() {
		a = new double[n][n];
		b = new double[n][n];
		c = new double[n][n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				a[i][j] = random.nextDouble();
				b[i][j] = random.nextDouble();
				c[i][j] = 0;
			}
		}
	}

	@Benchmark
	public void benchmarkMatrixMultiplication() {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < n; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
	@Benchmark
	public void matrixMultiplicationTransposed() {
		double[][] bt = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				bt[j][i] = b[i][j];
			}
		}

		for (int i = 0; i < n; i++) {
			double[] aRow = a[i];
			double[] cRow = c[i];
			for (int j = 0; j < n; j++) {
				double[] bRow = bt[j];
				double sum = 0;
				for (int k = 0; k < n; k++) {
					sum += aRow[k] * bRow[k];
				}
				cRow[j] = sum;
			}
		}
	}

}
