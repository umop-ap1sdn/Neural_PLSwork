package neural_plswork.math;

import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.exceptions.IllegalMatrixException;

public class IdentityMatrix extends Matrix<ConstantElement> {
    
    final int size;

    public IdentityMatrix(int size) throws IllegalMatrixException {
        super(Matrix.buildIdentityMatrix(size).matrix);
        this.size = size;
        
    }

    public <T extends MatrixElement> Vector<T> multiply(Vector<T> other) throws IllegalMatrixException {
        if(other.getRows() != size) throw new IllegalMatrixException("Matrices are incompatible for multiplication");
        return (Vector<T>) other.copy();
    }

    public int getSize() {
        return size;
    }
}
