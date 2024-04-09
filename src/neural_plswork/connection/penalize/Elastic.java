package neural_plswork.connection.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class Elastic implements Penalty {
    
	private final double lambda1;
	private final double lambda2;
	
	private final Lasso l1;
    private final Ridge l2;

    protected Elastic(double lambda1, double lambda2) {
        this.l1 = new Lasso(lambda1);
        this.l2 = new Ridge(lambda2);

		this.lambda1 = lambda1;
		this.lambda2 = lambda2;
    }

	@Override
	public synchronized Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights) {
		return l1.getPenalty(weights).add(l2.getPenalty(weights));
	}

	@Override
	public synchronized Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights) {
		return l1.getDerivative(weights).add(l2.getDerivative(weights));
	}

	@Override
	public Elastic copy() {
		return new Elastic(lambda1, lambda2);
	}
}
