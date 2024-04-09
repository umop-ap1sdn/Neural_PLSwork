package neural_plswork.connection.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class Lasso implements Penalty {
    private final double lambda;   
    
    protected Lasso(double lambda) {
        this.lambda = lambda;
    }

	@Override
	public synchronized Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights) {
		Matrix<NetworkValue> penalty = weights.copy();
        
        for(NetworkValue n: penalty) {
            n.setValue(lambda * Math.abs(n.getValue()));
        }

        return penalty;
	}

	@Override
	public synchronized Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights) {
		NetworkValue[][] derivative = new NetworkValue[weights.getRows()][weights.getColumns()];
        for(int i = 0; i < weights.getRows(); i++) {
            for(int j = 0; j < weights.getColumns(); j++) {
                double dir = 0;
                if(weights.getValue(i, j).getValue() > 0) dir = 1;
                if(weights.getValue(i, j).getValue() < 0) dir = -1;
                
                derivative[i][j] = new NetworkValue(lambda * dir);
            }
        }

        return new Matrix<NetworkValue>(derivative);
	}

    @Override
    public Lasso copy() {
        return new Lasso(lambda);
    }
}
