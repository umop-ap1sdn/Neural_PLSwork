package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Tanh implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            n.setValue(Math.tanh(n.getValue()));
        }

        return vector;
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            n.setValue(Math.pow(1 / Math.cosh(n.getValue()), 2));
        }

        return vector;
    }
    
}
