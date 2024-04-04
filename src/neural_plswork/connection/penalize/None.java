package neural_plswork.connection.penalize;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class None implements Penalty {

	@Override
	public Matrix<NetworkValue> getPenalty(Matrix<NetworkValue> weights) {
		NetworkValue[][] ret = new NetworkValue[weights.getRows()][weights.getColumns()];
        for(int i = 0; i < weights.getRows(); i++) {
            for(int j = 0; j < weights.getColumns(); j++) {
                ret[i][j] = new NetworkValue(0);
            }
        } 

        return new Matrix<>(ret);
	}

	@Override
	public Matrix<NetworkValue> getDerivative(Matrix<NetworkValue> weights) {
		NetworkValue[][] ret = new NetworkValue[weights.getRows()][weights.getColumns()];
        for(int i = 0; i < weights.getRows(); i++) {
            for(int j = 0; j < weights.getColumns(); j++) {
                ret[i][j] = new NetworkValue(0);
            }
        } 

        return new Matrix<>(ret);
	}

    @Override
    public None copy() {
        return new None();
    }
    
}
