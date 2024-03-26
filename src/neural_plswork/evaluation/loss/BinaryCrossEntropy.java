package neural_plswork.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.math.Vector;

public class BinaryCrossEntropy implements LossFunction, Differentiable {
    
    private static final double EPSILON = 1e-7;
    private final int batchSize;

    protected BinaryCrossEntropy(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public Vector<NetworkValue> calculateError(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        double[] targets = NetworkValue.vectorToArr(target);
        double[] predicts = NetworkValue.vectorToArr(predicted);

        double[] errors = new double[targets.length];
        
        for(int i = 0; i < targets.length; i++) {
            double oneErr = Math.log((predicts[i]) + EPSILON) / Math.log(2);
            double zeroErr = Math.log((1 - predicts[i]) + EPSILON) / Math.log(2);
            errors[i] = -1 * ((targets[i] * oneErr) + ((1 - targets[i]) * zeroErr)) / batchSize;
        }

        return NetworkValue.arrToVector(errors);
    }

    @Override
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        double[] targets = NetworkValue.vectorToArr(target);
        double[] predicts = NetworkValue.vectorToArr(predicted);

        double[] derivs = new double[targets.length];

        double divisor = batchSize * (Math.log(2) / Math.log(Math.E));
        
        for(int i = 0; i < targets.length; i++) {
            double oneDeriv = 1.0 / ((predicts[i] + EPSILON) * divisor);
            double zeroDeriv = -1.0 / (((1 - predicts[i]) + EPSILON) * divisor);
            derivs[i] = -1 * ((targets[i] * oneDeriv) + ((1 - targets[i]) * zeroDeriv));
        }

        return NetworkValue.arrToVector(derivs);
    }


}
