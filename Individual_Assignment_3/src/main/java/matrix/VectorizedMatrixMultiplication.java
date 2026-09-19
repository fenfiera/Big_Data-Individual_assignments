package matrix;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;

public class VectorizedMatrixMultiplication {

    public static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        double[][] BT = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                BT[j][i] = B[i][j];

        var species = DoubleVector.SPECIES_PREFERRED;

        for (int i = 0; i < n; i++) {
            double[] aRow = A[i];
            double[] cRow = C[i];

            for (int j = 0; j < n; j++) {
                double[] bRow = BT[j];
                double sum = 0;

                int k = 0;
                for (; k < species.loopBound(n); k += species.length()) {
                    var va = DoubleVector.fromArray(species, aRow, k);
                    var vb = DoubleVector.fromArray(species, bRow, k);
                    sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
                }

                for (; k < n; k++) {
                    sum += aRow[k] * bRow[k];
                }
                cRow[j] = sum;
            }
        }
        return C;
    }
}
