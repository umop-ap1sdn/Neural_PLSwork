package neural_plswork.activations;

import java.util.Arrays;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Linear implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        return unactivated.copy().getAsVector();
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        double[] ret = new double[unactivated.getLength()];
        Arrays.fill(ret, 1.0);
        return NetworkValue.arrToVector(ret);
    }
    
}
