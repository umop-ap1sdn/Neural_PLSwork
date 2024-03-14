package neural_plswork.math;

import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.exceptions.IllegalMatrixException;

public class IdentityMatrix extends Matrix<ConstantElement> {
    
    public IdentityMatrix(int size) throws IllegalMatrixException {
        super(Matrix.buildIdentityMatrix(size).matrix);
        
    }

    public <T extends MatrixElement> Vector<T> multiply(Vector<T> other) throws IllegalMatrixException {
        return other.copy();
    }
}
