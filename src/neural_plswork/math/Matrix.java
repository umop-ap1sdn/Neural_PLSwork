package neural_plswork.math;

import java.util.Arrays;

import neural_plswork.math.constants.AdditiveIdentity;
import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.constants.IdentityElement;
import neural_plswork.math.constants.MultiplicativeIdentity;
import neural_plswork.math.exceptions.IllegalMatrixException;
import neural_plswork.math.exceptions.IllegalVectorException;

public class Matrix<T extends MatrixElement> {
    
    private final int rows;
    private final int columns;

    protected final T[][] matrix;

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

    public Matrix(T[][] matrix) throws IllegalMatrixException {
        if(matrix.length == 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");
        if(matrix[0].length == 0) throw new IllegalArgumentException("Cannot instantiate matrix with a dimension less than 1");
        
        this.matrix = matrix;

        this.rows = matrix.length;
        this.columns = matrix[0].length;

        for(MatrixElement[] arr: matrix) {
            if(arr.length != columns) throw new IllegalMatrixException("Cannot instantiate non-rectangular matrix");
        }
    }

    public static Matrix<ConstantElement> buildIdentityMatrix(int size) throws IllegalMatrixException {
        ConstantElement[][] matrix = new ConstantElement[size][size];
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                if(i == j) matrix[i][j] = new MultiplicativeIdentity();
                else matrix[i][j] = new AdditiveIdentity();
            }
        }

        return new Matrix<ConstantElement>(matrix);
    }

    public <U extends MatrixElement> Matrix<U> multiply(Matrix<U> other) throws IllegalMatrixException {
        if(other.rows != this.columns) throw new IllegalMatrixException("Columns of matrix 0 does not match rows of matrix 1");
        Matrix<U> ret = new Matrix<>(this.rows, other.columns);
        
        for(int i = 0; i < ret.rows; i++) {
            for(int j = 0; j < ret.columns; j++) {
                ret.matrix[i][j] = multiplyAtLocation(other, i, j);
            }
        }
        
        return ret;
    }

    @SuppressWarnings("unchecked")
    private <U extends MatrixElement> U multiplyAtLocation(Matrix<U> other, int row, int col) {
        MatrixElement sum = new IdentityElement();
        
        for(int k = 0; k < this.columns; k++) {
            MatrixElement adder = matrix[row][k].multiply(other.matrix[k][col]);
            sum = sum.add(adder);
        }

        return (U) sum;
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
    public Matrix<T> pointwiseMultiply(Matrix<T> other) throws IllegalMatrixException {
        if(this.rows != other.rows || this.columns != other.columns) throw new IllegalMatrixException("Matrices must be of the same size to be added");
        
        Matrix<T> ret = new Matrix<>(this.rows, this.columns);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (T) matrix[i][j].multiply(other.matrix[i][j]);
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

    public Vector<T> getAsVector() throws IllegalVectorException {
        try {
            return new Vector<T>(matrix);
            
        } catch (IllegalMatrixException m) {
            System.out.println("an error occured.");
            System.out.println(m.getMessage());
            return null;
        }
    }

    public void simplePrint() {
        for(T[] arr: matrix) {
            System.out.println(Arrays.toString(arr));
        }
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
}
