package neural_plswork.evaluation.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface LossFunction {
    
    public Vector<NetworkValue> calculateError(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted);

    public static NetworkValue calculateOverallLoss(Vector<NetworkValue> errors) {
        double sum = 0;
        for(NetworkValue n: errors) sum += n.getValue();
        return new NetworkValue(sum);
    }
}
