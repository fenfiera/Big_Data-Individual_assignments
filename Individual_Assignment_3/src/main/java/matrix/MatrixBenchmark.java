package matrix;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class MatrixBenchmark {

    @Param({"128", "192", "384", "512", "1024"})
    private int n;

    private double[][] A;
    private double[][] B;

    private Random random = new Random(12345);

    @Setup(Level.Trial)
    public void setUp() {
        A = new double[n][n];
        B = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = random.nextDouble();
                B[i][j] = random.nextDouble();
            }
        }
    }

    @Benchmark
    public void naive(Blackhole bh) {
        double[][] C = NaiveMatrixMultiplication.multiply(A, B);
        bh.consume(C);
    }

    @Benchmark
    public void vectorized(Blackhole bh) {
        double[][] C = VectorizedMatrixMultiplication.multiply(A, B);
        bh.consume(C);
    }

    @Benchmark
    public void parallel(Blackhole bh) {
        double[][] C = ParallelMatrixMultiplication.multiply(A, B);
        bh.consume(C);
    }
}
