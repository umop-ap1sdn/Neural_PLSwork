package neural_plswork.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.math.Vector;
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

        Matrix<NetworkValue> intermediate = predicted.scale(new NetworkValue(-1));
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

        Matrix<NetworkValue> intermediate = target.scale(new NetworkValue(-1));
        intermediate = intermediate.add(predicted);
        intermediate.scale(new NetworkValue(2.0 / batchSize));
        return intermediate.getAsVector();

    }
}
