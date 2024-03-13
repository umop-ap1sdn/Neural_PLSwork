package neural_plswork.math.constants;

import neural_plswork.math.MatrixElement;

public class AdditiveIdentity extends IdentityElement {
    @Override
    public MatrixElement multiply(MatrixElement other) {
        return this;
    }
}
