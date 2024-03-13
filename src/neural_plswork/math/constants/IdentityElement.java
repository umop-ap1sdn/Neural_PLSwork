package neural_plswork.math.constants;

import neural_plswork.math.MatrixElement;

public class IdentityElement implements ConstantElement {

    @Override
    public MatrixElement multiply(MatrixElement other) {
        return other;
    }

    @Override
    public MatrixElement add(MatrixElement other) {
        return other;
    }

    @Override
    public MatrixElement negate() {
        return this;
    }

    @Override
    public MatrixElement copy() {
        return new IdentityElement();
    }
    
    
}
