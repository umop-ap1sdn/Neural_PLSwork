package neural_plswork.neuron.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Differentiable;

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

    @Override
    public double calculateEval(double[][] y_true, double[][] y_pred) {
        if(y_true.length != y_pred.length) throw new IllegalArgumentException("Input arrays must be of the same size");
        
        double errSum = 0;
        for(int i = 0; i < y_true.length; i++) {
            double subSum = 0;
            for(int j = 0; j < y_true[i].length; j++) {
                if(y_pred[i].length != y_true[i].length) throw new IllegalArgumentException("Input arrays must be of the same size");
                double oneErr = Math.log(y_pred[i][j] + EPSILON);
                subSum += -1 * (y_true[i][j] * oneErr);
            }

            errSum += subSum;
        }

        return errSum / y_true.length;
    }
    
}
