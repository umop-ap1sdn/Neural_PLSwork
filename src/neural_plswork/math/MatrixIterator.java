package neural_plswork.math;

import java.util.Iterator;

public class MatrixIterator<T extends MatrixElement> implements Iterator<T> {

    private final Matrix<T> iterate;
    private int row;
    private int col;

    public MatrixIterator(Matrix<T> iterate) {
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
