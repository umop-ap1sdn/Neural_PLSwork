package neural_plswork.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.math.Vector;

public class CrossEntropy implements LossFunction, Differentiable {

    private static final double EPSILON = 1e-7;
    private final int batchSize;

    protected CrossEntropy(int batchSize) {
        this.batchSize = batchSize;
    }


    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        NetworkValue[] errors = new NetworkValue[target.getLength()];

        for(int i = 0; i < errors.length; i++) {
            double value = -1 * target.getValue(i).getValue() * Math.log(predicted.getValue(i).getValue() + EPSILON) / batchSize;
            errors[i] = new NetworkValue(value);
        }

        return new Vector<>(errors);
    }

    @Override
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        NetworkValue[] derivs = new NetworkValue[target.getLength()];

        for(int i = 0; i < derivs.length; i++) {
            double value = -1 * target.getValue(i).getValue() / ((predicted.getValue(i).getValue() + EPSILON) * batchSize);
            derivs[i] = new NetworkValue(value);
        }

        return new Vector<>(derivs);
    }
    
}
