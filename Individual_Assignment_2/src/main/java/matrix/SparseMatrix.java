package matrix;

import java.util.HashMap;
import java.util.Map;

public class SparseMatrix {
    private final Map<Integer, Map<Integer, Double>> values = new HashMap<>();

    public void put(int row, int col, double val) {
        if (val != 0.0) {
            values.computeIfAbsent(row, r -> new HashMap<>()).put(col, val);
        }
    }

    public Map<Integer, Double> getRow(int row) {
        return values.getOrDefault(row, new HashMap<>());
    }

    public static SparseMatrix multiplySparse(SparseMatrix A, SparseMatrix B) {
        SparseMatrix C = new SparseMatrix();
        for (int i : A.values.keySet()) {
            Map<Integer, Double> rowA = A.getRow(i);
            for (int k : rowA.keySet()) {
                double aVal = rowA.get(k);
                Map<Integer, Double> rowB = B.getRow(k);
                for (int j : rowB.keySet()) {
                    double bVal = rowB.get(j);
                    double prev = C.getRow(i).getOrDefault(j, 0.0);
                    C.put(i, j, prev + aVal * bVal);
                }
            }
        }
        return C;
    }
}