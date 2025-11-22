package matrix;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import java.util.concurrent.TimeUnit;
import java.util.Random;


@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
public class MatrixMultiplication {

    @Param({"128", "512", "1024"})
    private int n;

    @Param({"0", "25", "50", "75", "90"})
    private int sparsity;

    private double[][] a, b, c;
    private SparseMatrix sa, sb;
    private Random random = new Random();

    @Setup(Level.Invocation)
    public void setUp() {
        a = new double[n][n];
        b = new double[n][n];
        c = new double[n][n];
        sa = new SparseMatrix();
        sb = new SparseMatrix();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (random.nextInt(100) < sparsity) {
                    a[i][j] = 0.0;
                    b[i][j] = 0.0;
                } else {
                    a[i][j] = random.nextDouble();
                    b[i][j] = random.nextDouble();
                    sa.put(i, j, a[i][j]);
                    sb.put(i, j, b[i][j]);
                }
                c[i][j] = 0.0;
            }
        }
    }

    // ---------------- Benchmarks ----------------

    @Benchmark
    public void MatrixMultiplicationNaive(Blackhole bh) {
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        bh.consume(result);
    }

    @Benchmark
    public void MatrixMultiplicationStrassenOptimized(Blackhole bh) {
        double[][] result = StrassenOptimized.multiply(a, b);
        bh.consume(result);
    }

    @Benchmark
    public void MatrixMultiplicationSparse(Blackhole bh) {
        SparseMatrix result = SparseMatrix.multiplySparse(sa, sb);
        bh.consume(result);
    }
}
