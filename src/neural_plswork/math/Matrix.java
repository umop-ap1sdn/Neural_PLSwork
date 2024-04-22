package neural_plswork.math;

import java.util.Arrays;
import java.util.Iterator;

import neural_plswork.math.constants.AdditiveIdentity;
import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.constants.IdentityElement;
import neural_plswork.math.constants.MultiplicativeIdentity;
import neural_plswork.math.exceptions.ElementIncompatibleException;
import neural_plswork.math.exceptions.IllegalMatrixException;
import neural_plswork.math.exceptions.IllegalVectorException;

public class Matrix<T extends MatrixElement> implements Iterable<T> {
    
    protected final int rows;
    protected final int columns;

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

    protected Matrix(T[][] matrix, boolean override) throws IllegalMatrixException {
        this.matrix = matrix;

        this.rows = matrix.length;
        this.columns = matrix[0].length;
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

    public <U extends MatrixElement, V extends MatrixElement> Matrix<U> multiply(Matrix<V> other) throws IllegalMatrixException, ElementIncompatibleException {
        if(other.rows != this.columns) throw new IllegalMatrixException("Columns of matrix 0 does not match rows of matrix 1");
        if(!matrix[0][0].addable(other.matrix[0][0])) throw new ElementIncompatibleException("Cannot add these 2 types");
        if(!matrix[0][0].multipliable(other.matrix[0][0])) throw new ElementIncompatibleException("Cannot multiply these 2 types");



        Matrix<U> ret = new Matrix<U>(this.rows, other.columns);
        
        for(int i = 0; i < ret.rows; i++) {
            for(int j = 0; j < ret.columns; j++) {
                ret.matrix[i][j] = multiplyAtLocation(other, i, j);
            }
        }
        
        return ret;
    }

    @SuppressWarnings("unchecked")
    private <U extends MatrixElement, V extends MatrixElement> U multiplyAtLocation(Matrix<V> other, int row, int col) {
        MatrixElement sum = new IdentityElement();
        
        for(int k = 0; k < this.columns; k++) {
            MatrixElement adder = matrix[row][k].multiply(other.matrix[k][col]);
            sum = sum.add(adder);
        }

        return (U) sum;
    }

    @SuppressWarnings("unchecked")
    public <U extends MatrixElement, V extends MatrixElement> Matrix<U> add(Matrix<V> other) throws IllegalMatrixException, ElementIncompatibleException {
        if(this.rows != other.rows || this.columns != other.columns) throw new IllegalMatrixException("Matrices must be of the same size to be added");
        if(!matrix[0][0].addable(other.matrix[0][0])) throw new ElementIncompatibleException("Cannot add these 2 types");
        

        Matrix<U> ret = new Matrix<>(this.rows, this.columns);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (U) matrix[i][j].add(other.matrix[i][j]);
            }
        }

        return ret;
    }

    @SuppressWarnings("unchecked")
    public <U extends MatrixElement, V extends MatrixElement> Matrix<U> pointwiseMultiply(Matrix<V> other) throws IllegalMatrixException, ElementIncompatibleException {
        if(this.rows != other.rows || this.columns != other.columns) throw new IllegalMatrixException("Matrices must be of the same size to be pointwise multiplied");
        if(!matrix[0][0].multipliable(other.matrix[0][0])) throw new ElementIncompatibleException("Cannot multiply these 2 types");

        Matrix<U> ret = new Matrix<>(this.rows, this.columns);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (U) matrix[i][j].multiply(other.matrix[i][j]);
            }
        }

        return ret;
    }

    @SuppressWarnings("unchecked")
    public <U extends MatrixElement, V extends MatrixElement> Matrix<U> scale(V other) {
        if(!other.multipliable(matrix[0][0])) throw new ElementIncompatibleException("Cannot multiply these 2 types");
        Matrix<U> ret = new Matrix<>(this.rows, this.columns);
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                ret.matrix[i][j] = (U) other.multiply(matrix[i][j]);
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
        if(columns != 1) throw new IllegalVectorException("Vector must have a column length of 1");
        return new Vector<T>(matrix, true);
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

    @Override
    public Iterator<T> iterator() {
        return new MatrixIterator(this);
    }
    
    class MatrixIterator implements Iterator<T> {

        private final Matrix<T> iterate;
        private int row;
        private int col;
    
        private MatrixIterator(Matrix<T> iterate) {
            this.iterate = iterate;
            row = 0;
            col = 0;
        }
    
        @Override
        public boolean hasNext() {
            if(row == iterate.rows) return false;
            return true;
        }
    
        @Override
        public T next() {
            T next = iterate.getValue(row, col);
            col = (col + 1) % iterate.columns;
            if(col == 0) row++;
            return next;
        }
        
    }
}
