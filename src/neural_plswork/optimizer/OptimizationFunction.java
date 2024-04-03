package neural_plswork.optimizer;

import neural_plswork.core.Copiable;
import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;

public interface OptimizationFunction extends Copiable {
    public Matrix<NetworkValue> computeDeltas(Matrix<NetworkValue> gradients, double learning_rate);
    public OptimizationFunction copy();
    
    public static OptimizationFunction getFunction(Optimizer optimizer) throws InvalidOptimizerException {
        if(optimizer == null) throw new InvalidOptimizerException("Optimizer enum is null");
        
        switch(optimizer) {
            case CUSTOM: return null;
            case SGD: return new StochasticGradientDescent();
            case ADAM: return new Adam();
            case INVALID: throw new InvalidOptimizerException();
            default: throw new InvalidOptimizerException();
        }
    }
}
