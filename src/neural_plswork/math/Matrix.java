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

/**
 * The matrix class is used to facilitate the linear algebra involved in the training and testing of neural networks.<br>
 * This class allows matrices to be added, multiplied, pointwise-multiplied, and scaled. Additionally, matrices are able to be configured
 * with any class which extends MatrixElement to allow for general usage.
 */

public class Matrix<T extends MatrixElement> implements Iterable<T> {
    
    protected final int rows;
    protected final int columns;

    protected final T[][] matrix;

    /**
     * Basic constructor which creates an empty matrix of the specified dimensions
     * @param rows Rows in the matrix
     * @param columns Columns in the matrix
     * @throws IllegalArgumentException If either rows or columns are < 1
     */
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

    /**
     * Constructor to create a matrix from a 2D array
     * @param matrix 2D source array
     * @throws IllegalMatrixException if any length of the array is < 1 or array is non-rectangular
     */
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

    /**
     * Function to build an identity matrix of the specified size; an identity matrix consists of a square matrix with 
     * the multiplicative identity on primary diagonal and additive identities elsewhere.
     * @param size Size of the matrix
     * @return Identity matrix of the given size
     * @throws IllegalMatrixException if the given size does not match with the size requirements
     */
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

    /**
     * Function to multiply 2 matrices
     * @param <U> Type of the other matrix
     * @param <V> Type of the product matrix
     * @param other matrix to be multiplied with
     * @return product matrix
     * @throws IllegalMatrixException If the factor matricies do not have a compatible size, IllegalMatrixException will be thrown <br>
     * Number of columns in the first matrix must match number of rows in the second matrix
     * @throws ElementIncompatibleException If the elements are incompatible, ElementIncompatibleException will be thrown
     */
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

    /**
     * Multiply helper function to complete a step of multiplication at a single spot in the matrix
     * @param <U> Type of the other matrix
     * @param <V> Type of the resultant matrix
     * @param other Matrix being multiplied with
     * @param row Row being multiplied on
     * @param col Column being multiplied on
     * @return Product along the row, column specified
     */
    @SuppressWarnings("unchecked")
    private <U extends MatrixElement, V extends MatrixElement> U multiplyAtLocation(Matrix<V> other, int row, int col) {
        MatrixElement sum = new IdentityElement();
        
        for(int k = 0; k < this.columns; k++) {
            MatrixElement adder = matrix[row][k].multiply(other.matrix[k][col]);
            sum = sum.add(adder);
        }

        return (U) sum;
    }

    /**
     * Function to add 2 matrices
     * @param <U> Type of the other matrix
     * @param <V> Type of the sum matrix
     * @param other matrix to be added with
     * @return sum matrix
     * @throws IllegalMatrixException If the factor matricies do not have a compatible size, IllegalMatrixException will be thrown <br>
     * Number of rows and columns of both matrices must match
     * @throws ElementIncompatibleException If the elements are incompatible, ElementIncompatibleException will be thrown
     */
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

    /**
     * Function to pointwise multiply 2 matrices<br>
     * Pointwise multiplying matrices involves multiplying each point of the first with the respective point of the second matrix, similar to matrix addition
     * @param <U> Type of the other matrix
     * @param <V> Type of the product matrix
     * @param other matrix to be multiplied with
     * @return product matrix
     * @throws IllegalMatrixException If the factor matricies do not have a compatible size, IllegalMatrixException will be thrown <br>
     * Number of rows and columns of both matrices must match
     * @throws ElementIncompatibleException If the elements are incompatible, ElementIncompatibleException will be thrown
     */
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

    /**
     * Function to scale a matrix by a multipliable factor
     * @param <U> Resultant Matrix type
     * @param <V> Factor type
     * @param other Factor to scale the matrix with
     * @return Scaled matrix
     */
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

    /**
     * Function to transpose a matrix; swapping rows with columns
     * @return Transposed matrix
     */
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
     /**
      * Function to make a deep copy of the matrix
      * @return copied matrix
      */
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

    /**
     * Function to set a value of a particular coordinate point of a matrix
     * @param value Value to set
     * @param row row coordinate
     * @param col column coordinate
     * @throws ArrayIndexOutOfBoundsException thrown if either row or column is out of bound of the matrix
     */
    public void setValue(T value, int row, int col) throws ArrayIndexOutOfBoundsException {
        if(row < 0 || row >= rows) throw new ArrayIndexOutOfBoundsException("Matrix row index out of bounds");
        if(col < 0 || col >= columns) throw new ArrayIndexOutOfBoundsException("Matrix column index out of bounds");

        matrix[row][col] = value;
    }

    /**
     * Function to get a value of a particular coordinate point of a matrix
     * @param row row coordinate
     * @param col column coordinate
     * @return Value at the specified coordinate point
     * @throws ArrayIndexOutOfBoundsException thrown if either row or column is out of bound of the matrix
     */
    public T getValue(int row, int col) throws ArrayIndexOutOfBoundsException {
        if(row < 0 || row >= rows) throw new ArrayIndexOutOfBoundsException("Matrix row index out of bounds");
        if(col < 0 || col >= columns) throw new ArrayIndexOutOfBoundsException("Matrix column index out of bounds");

        return matrix[row][col];
    }

    /**
     * Function to specify the Matrix as a vector (only applies to single column matrices)
     * @return Vector form of the matrix
     * @throws IllegalVectorException thrown if the matrix has more than one column
     */
    public Vector<T> getAsVector() throws IllegalVectorException {
        if(columns != 1) throw new IllegalVectorException("Vector must have a column length of 1");
        return new Vector<T>(matrix, true);
    }

    /**
     * Function to print the matrix 
     */
    public void simplePrint() {
        for(T[] arr: matrix) {
            System.out.println(Arrays.toString(arr));
        }
    }

    /**
     * Function to get the rows in the matrix
     * @return row count
     */
    public int getRows() {
        return rows;
    }

    /**
     * Function to get the columns in the matrix
     * @return column count
     */
    public int getColumns() {
        return columns;
    }

    @Override
    public Iterator<T> iterator() {
        return new MatrixIterator(this);
    }
    

    /**
     * Matrix iterator to support enhanced for loops
     */
    class MatrixIterator implements Iterator<T> {

        private final Matrix<T> iterate;
        private int row;
        private int col;
    
        /**
         * Primary constructor
         * @param iterate Matrix to be iterated
         */
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
