package neural_plswork.loss;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface LossFunction {
    
    public Vector<NetworkValue> calculateError(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
    
}
