package neural_plswork.activations;

import neural_plswork.math.Vector;
import neural_plswork.core.NetworkValue;

public interface ActivationFunction {
    
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated);
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated);
}
