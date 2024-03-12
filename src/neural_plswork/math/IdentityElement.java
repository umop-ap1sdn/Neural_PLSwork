package neural_plswork.math;

public class IdentityElement implements MatrixElement {

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
