package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Sigmoid implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            n.setValue(sigmoid(n.getValue()));
        }

        return vector;
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            n.setValue(sigmoid(n.getValue()) * (1 - sigmoid(n.getValue())));
        }

        return vector;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-1 * x));
    }
    
}
