package neural_plswork.neuron.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.math.Matrix;


public class MeanSquaredError implements LossFunction, Differentiable {
    private final int batchSize;    

    protected MeanSquaredError(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        /* 
        double[] targets = NetworkValue.vectorToArr(target);
        double[] predicts = NetworkValue.vectorToArr(predicted);

        double[] errors = new double[targets.length];
        
        for(int i = 0; i < targets.length; i++) {
            errors[i] = Math.pow(targets[i] - predicts[i], 2) / batchSize;
        }

        return NetworkValue.arrToVector(errors);
        */

        Matrix<NetworkValue> intermediate = predicted.scale(new NetworkValue(-1.0));
        intermediate = intermediate.add(target);
        intermediate = intermediate.pointwiseMultiply(intermediate);
        intermediate.scale(new NetworkValue(1.0 / batchSize));
        return intermediate.getAsVector();
    }

    @Override
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        /*
        double[] targets = NetworkValue.vectorToArr(target);
        double[] predicts = NetworkValue.vectorToArr(predicted);
        double[] derivs = new double[predicts.length];

        for(int i = 0; i < targets.length; i++) {
            derivs[i] = 2 * (predicts[i] - targets[i]) / batchSize;
        }

        return NetworkValue.arrToVector(derivs);

        */

        Matrix<NetworkValue> intermediate = target.scale(new NetworkValue(-1.0));
        intermediate = intermediate.add(predicted);
        intermediate.scale(new NetworkValue(2.0 / batchSize));
        return intermediate.getAsVector();
    }

    @Override
    public double calculateEval(double[][] y_true, double[][] y_pred) {
        if(y_true.length != y_pred.length) throw new IllegalArgumentException("Input arrays must be of the same size");
        
        double errSum = 0;
        for(int i = 0; i < y_true.length; i++) {
            double subSum = 0;
            for(int j = 0; j < y_true[i].length; j++) {
                if(y_pred[i].length != y_true[i].length) throw new IllegalArgumentException("Input arrays must be of the same size");
                subSum += Math.pow((y_true[i][j] - y_pred[i][j]), 2);
            }

            errSum += subSum;
        }

        return errSum / y_true.length;
    }
}
