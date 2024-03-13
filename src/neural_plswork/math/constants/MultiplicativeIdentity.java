package neural_plswork.math.constants;

import neural_plswork.math.MatrixElement;

public class MultiplicativeIdentity extends IdentityElement {
    @Override
    public MatrixElement add(MatrixElement other) {
        return this;
    }
}
