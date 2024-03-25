package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class ReLU implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            n.setValue(Math.max(n.getValue(), 0.0));
        }

        return vector;
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        for(NetworkValue n: vector) {
            if(n.getValue() <= 0) n.setValue(0.0);
            else n.setValue(1.0);
        }

        return vector;
    }
    
}
