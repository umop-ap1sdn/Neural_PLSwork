package neural_plswork.optimizer;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public class StochasticGradientDescent implements OptimizationFunction {

	@Override
	public Matrix<NetworkValue> computeDeltas(Matrix<NetworkValue> gradients, double learning_rate) {
		return gradients.scale(new NetworkValue(learning_rate));
	}
    
	@Override
	public StochasticGradientDescent copy() {
		return new StochasticGradientDescent();
	}
}
