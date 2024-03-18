package neural_plswork.math.constants;

import neural_plswork.math.MatrixElement;

public class NullElement implements ConstantElement {

    @Override
    public MatrixElement multiply(MatrixElement other) {
        return this;
    }

    @Override
    public MatrixElement add(MatrixElement other) {
        return this;
    }

    @Override
    public MatrixElement negate() {
        return this;
    }

    @Override
    public MatrixElement copy() {
        return new NullElement();
    }

    @Override
    public boolean addable(MatrixElement other) {
        return true;
    }

    @Override
    public boolean multipliable(MatrixElement other) {
        return true;
    }
    
}
