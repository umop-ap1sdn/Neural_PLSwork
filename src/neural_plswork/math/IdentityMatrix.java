package neural_plswork.math;

import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.exceptions.IllegalMatrixException;

public class IdentityMatrix {
    final Matrix<ConstantElement> matrix;

    public IdentityMatrix(int size) throws IllegalMatrixException {
        matrix = Matrix.buildIdentityMatrix(size);
        
    }

    public <T extends MatrixElement> Matrix<T> multiply(Vector<T> other) {
        return other.copy();
    }
}
