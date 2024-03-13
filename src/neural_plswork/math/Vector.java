package neural_plswork.math;

import neural_plswork.math.exceptions.IllegalMatrixException;
import neural_plswork.math.exceptions.IllegalVectorException;

public class Vector<T extends MatrixElement> extends Matrix<T> {
    public Vector(int rows) {
        super(rows, 1);
    }

    @SuppressWarnings("unchecked")
    public Vector(T[] oneD) throws IllegalMatrixException {
        super((T[][]) transposeTo2d(oneD));
    }

    public Vector(T[][] vector) throws IllegalVectorException, IllegalMatrixException {
        super(vector);
        if(vector[0].length != 1) throw new IllegalVectorException("Vector must have a column length of 1");
    }
    
    private static MatrixElement[][] transposeTo2d(MatrixElement[] oneD) {
        MatrixElement[][] ret = new MatrixElement[oneD.length][1];
        for(int i = 0; i < oneD.length; i++) {
            ret[i][0] = oneD[i].copy();
        }

        return ret;
    }

    @SuppressWarnings("unchecked")
    public T[] get1D() {
        MatrixElement[] ret = new MatrixElement[matrix.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = matrix[i][0];
        }

        return (T[]) ret;
    }
}
