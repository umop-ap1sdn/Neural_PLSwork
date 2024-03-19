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
    public MatrixElement copy() {
        return new IdentityElement();
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
