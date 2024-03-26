package neural_plswork.evaluation;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public interface Differentiable {
    public Vector<NetworkValue> calculateDerivative(Vector<NetworkValue> target, Vector<NetworkValue> predicted);
}
