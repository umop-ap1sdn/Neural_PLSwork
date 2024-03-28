package neural_plswork.optimizer;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public interface OptimizationFunction {
    public Matrix<NetworkValue> computeDeltas(Matrix<NetworkValue> gradients, double learning_rate);
    public OptimizationFunction copy();
}
