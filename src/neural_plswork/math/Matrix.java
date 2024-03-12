package neural_plswork.math;

import java.util.Arrays;

public class Matrix<T extends MatrixElement> {
    
    private final int rows;
    private final int columns;

    private final T[][] matrix;

    @SuppressWarnings("unchecked")
    public Matrix(int rows, int columns) throws IllegalArgumentException {
        if(rows <= 0 || columns <= 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");

        this.rows = rows;
        this.columns = columns;

        this.matrix = (T[][]) new MatrixElement[rows][columns];

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                matrix[i][j] = null;
            }
        }
    }

    public Matrix(T[][] matrix) throws Exception {
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
        if(other.rows != this.columns) throw new IllegalMatrixException("Columns of matrix 0 does not match rows of matrix 1");
        Matrix<T> ret = new Matrix<>(this.rows, other.columns);
        
        for(int i = 0; i < ret.rows; i++) {
            for(int j = 0; j < ret.columns; j++) {
                ret.matrix[i][j] = multiplyAtLocation(other, i, j);
            }
        }
        
        return ret;
    }

    @SuppressWarnings("unchecked")
    private T multiplyAtLocation(Matrix<T> other, int row, int col) {
        MatrixElement sum = new IdentityElement();
        
        for(int k = 0; k < this.columns; k++) {
            MatrixElement adder = matrix[row][k].multiply(matrix[k][col]);
            sum = sum.add(adder);
        }

        return (T) sum;
    }

    @SuppressWarnings("unchecked")
    public Matrix<T> add(Matrix<T> other) throws IllegalMatrixException {
        if(this.rows != other.rows || this.columns != other.columns) throw new IllegalMatrixException("Matrices must be of the same size to be added");
        
        Matrix<T> ret = new Matrix<>(this.rows, this.columns);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (T) matrix[i][j].add(other.matrix[i][j]);
            }
        }

        return ret;
    }

    @SuppressWarnings("unchecked")
    public Matrix<T> transpose() {
        Matrix<T> ret = new Matrix<>(columns, rows);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[j][i] = (T) matrix[i][j].copy();
            }
        }
        
        return ret;
    }

    @SuppressWarnings("unchecked")
    public Matrix<T> negate() {
        Matrix<T> ret = new Matrix<>(rows, columns);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (T) matrix[i][j].negate();
            }
        }

        return ret;
    }

    @SuppressWarnings("unchecked")
    public Matrix<T> copy() {
        Matrix<T> ret = new Matrix<>(rows, columns);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (T) matrix[i][j].copy();
            }
        }

        return ret;
    }

    public void setValue(T value, int row, int col) throws ArrayIndexOutOfBoundsException {
        if(row < 0 || row >= rows) throw new ArrayIndexOutOfBoundsException("Matrix row index out of bounds");
        if(col < 0 || col >= columns) throw new ArrayIndexOutOfBoundsException("Matrix column index out of bounds");

        matrix[row][col] = value;
    }

    public T getValue(int row, int col) throws ArrayIndexOutOfBoundsException {
        if(row < 0 || row >= rows) throw new ArrayIndexOutOfBoundsException("Matrix row index out of bounds");
        if(col < 0 || col >= columns) throw new ArrayIndexOutOfBoundsException("Matrix column index out of bounds");

        return matrix[row][col];
    }


    public void simplePrint() {
        for(T[] arr: matrix) {
            System.out.println(Arrays.toString(arr));
        }
    }
}
