package matrix;

public class StrassenOptimized {
    private static final int CUTOFF = 64;

    public static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        if (n <= CUTOFF) {
            return multiplyBlockedUnrolled(A, B);
        }

        int newSize = n / 2;
        double[][] a11 = new double[newSize][newSize];
        double[][] a12 = new double[newSize][newSize];
        double[][] a21 = new double[newSize][newSize];
        double[][] a22 = new double[newSize][newSize];

        double[][] b11 = new double[newSize][newSize];
        double[][] b12 = new double[newSize][newSize];
        double[][] b21 = new double[newSize][newSize];
        double[][] b22 = new double[newSize][newSize];

        split(A, a11, 0 , 0);
        split(A, a12, 0 , newSize);
        split(A, a21, newSize, 0);
        split(A, a22, newSize, newSize);

        split(B, b11, 0 , 0);
        split(B, b12, 0 , newSize);
        split(B, b21, newSize, 0);
        split(B, b22, newSize, newSize);

        double[][] m1 = multiply(add(a11, a22), add(b11, b22));
        double[][] m2 = multiply(add(a21, a22), b11);
        double[][] m3 = multiply(a11, subtract(b12, b22));
        double[][] m4 = multiply(a22, subtract(b21, b11));
        double[][] m5 = multiply(add(a11, a12), b22);
        double[][] m6 = multiply(subtract(a21, a11), add(b11, b12));
        double[][] m7 = multiply(subtract(a12, a22), add(b21, b22));

        double[][] c11 = add(subtract(add(m1, m4), m5), m7);
        double[][] c12 = add(m3, m5);
        double[][] c21 = add(m2, m4);
        double[][] c22 = add(subtract(add(m1, m3), m2), m6);

        double[][] C = new double[n][n];
        join(c11, C, 0 , 0);
        join(c12, C, 0 , newSize);
        join(c21, C, newSize, 0);
        join(c22, C, newSize, newSize);

        return C;
    }

    private static double[][] add(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    private static double[][] subtract(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    private static void split(double[][] P, double[][] C, int iB, int jB) {
        for (int i = 0; i < C.length; i++)
            for (int j = 0; j < C.length; j++)
                C[i][j] = P[i + iB][j + jB];
    }

    private static void join(double[][] C, double[][] P, int iB, int jB) {
        for (int i = 0; i < C.length; i++)
            for (int j = 0; j < C.length; j++)
                P[i + iB][j + jB] = C[i][j];
    }

    public static double[][] multiplyBlockedUnrolled(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        int blockSize = 64;

        for (int ii = 0; ii < n; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {
                    for (int i = ii; i < Math.min(ii+blockSize, n); i++) {
                        for (int j = jj; j < Math.min(jj+blockSize, n); j++) {
                            double sum = 0;
                            for (int k = kk; k < Math.min(kk+blockSize, n); k += 4) {
                                sum += A[i][k] * B[k][j];
                                if (k+1 < n) sum += A[i][k+1] * B[k+1][j];
                                if (k+2 < n) sum += A[i][k+2] * B[k+2][j];
                                if (k+3 < n) sum += A[i][k+3] * B[k+3][j];
                            }
                            C[i][j] += sum;
                        }
                    }
                }
            }
        }
        return C;
    }
}