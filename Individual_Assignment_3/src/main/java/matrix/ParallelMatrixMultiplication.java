package matrix;

import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class ParallelMatrixMultiplication {

    private static final int THRESHOLD = 32;

    public static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        double[][] B_T = transpose(B);
        ForkJoinPool.commonPool().invoke(new MultiplyTask(A, B_T, C, 0, n));
        return C;
    }


    private static double[][] transpose(double[][] M) {
        int n = M.length;
        double[][] T = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = M[i][j];
            }
        }
        return T;
    }

    static class MultiplyTask extends RecursiveAction {
        private final double[][] A, B_T, C;
        private final int start, end;

        MultiplyTask(double[][] A, double[][] B_T, double[][] C, int start, int end) {
            this.A = A;
            this.B_T = B_T;
            this.C = C;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int size = end - start;
            if (size <= THRESHOLD) {
                multiplyRange(A, B_T, C, start, end);
                return;
            }
            int mid = (start + end) / 2;

            MultiplyTask left = new MultiplyTask(A, B_T, C, start, mid);
            MultiplyTask right = new MultiplyTask(A, B_T, C, mid, end);

            left.fork();
            right.compute();
            left.join();
        }

        private static void multiplyRange(double[][] A, double[][] B_T, double[][] C,
                                          int start, int end) {
            int n = B_T.length;
            for (int i = start; i < end; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B_T[j][k];
                    }
                    C[i][j] = sum;
                }
            }
        }
    }
}
