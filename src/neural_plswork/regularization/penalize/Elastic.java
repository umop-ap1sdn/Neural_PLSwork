package neural_plswork.regularization.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class Elastic implements Penalty {
    private final Lasso l1;
    private final Ridge l2;

    protected Elastic(double lambda1, double lambda2) {
        this.l1 = new Lasso(lambda1);
        this.l2 = new Ridge(lambda2);
    }

	@Override
	public Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights) {
		return l1.getPenalty(weights).add(l2.getPenalty(weights));
	}

	@Override
	public Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights) {
		return l1.getDerivative(weights).add(l2.getDerivative(weights));
	}
}
