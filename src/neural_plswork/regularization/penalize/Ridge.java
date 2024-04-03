package neural_plswork.regularization.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class Ridge implements Penalty {
    private final double lambda;   
    
    protected Ridge(double lambda) {
        this.lambda = lambda;
    }

	@Override
	public Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights) {
		Matrix<NetworkValue> penalty = weights.copy();
        
        for(NetworkValue n: penalty) {
            n.setValue(lambda * Math.pow(n.getValue(), 2));
        }

        return penalty;
	}

	@Override
	public Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights) {
		Matrix<NetworkValue> derivative = weights.copy();
        
        for(NetworkValue n: derivative) {
            n.setValue(lambda * 2 * n.getValue());
        }

        return derivative;
	}

    @Override
    public Ridge copy() {
        return new Ridge(lambda);
    }
    
}
