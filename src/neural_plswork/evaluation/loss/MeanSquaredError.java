package neural_plswork.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class MeanSquaredError implements LossFunction {
    private final int batchSize;    

    public MeanSquaredError(int batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public Vector<NetworkValue> calculateError(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        double[] targets = NetworkValue.vectorToArr(target);
        double[] predicts = NetworkValue.vectorToArr(predicted);

        double[] errors = new double[targets.length];
        
        for(int i = 0; i < targets.length; i++) {
            errors[i] = Math.pow(targets[i] - predicts[i], 2) / batchSize;
        }

        return NetworkValue.arrToVector(errors);
    }

    @Override
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'calculateDerivative'");
    }
}
