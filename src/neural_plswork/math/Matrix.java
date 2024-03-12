package neural_plswork.math;

public class Matrix<T extends MatrixElement> {
    
    final int rows;
    final int columns;

    final MatrixElement[][] matrix;

    public Matrix(int rows, int columns) throws IllegalArgumentException {
        if(rows <= 0 || columns <= 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");

        this.rows = rows;
        this.columns = columns;

        this.matrix = new MatrixElement[rows][columns];
    }

    public Matrix(MatrixElement[][] matrix) throws Exception {
        if(matrix.length == 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");
        if(matrix[0].length == 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");
        
        this.matrix = matrix;

        this.rows = matrix.length;
        this.columns = matrix[0].length;

        for(MatrixElement[]arr: matrix) {
            if(arr.length != columns) throw new IllegalMatrixException("Cannot instantiate non-rectangular matrix");
        }
    }

    public Matrix<T> multiply(Matrix<T> other) throws IllegalMatrixException {
        return null;
    }

    public Matrix<T> add(Matrix<T> other) throws IllegalMatrixException {
        return null;
    }

    public Matrix<T> transpose(Matrix<T> other) {
        return null;
    }

    public Matrix<T> negate() {
        return null;
    }
}
