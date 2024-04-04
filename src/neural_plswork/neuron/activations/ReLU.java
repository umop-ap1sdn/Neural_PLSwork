package neural_plswork.neuron.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;
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
    public Matrix<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Matrix<NetworkValue> jacobian = new Matrix<>(unactivated.getLength(), unactivated.getLength());
        for(int i = 0; i < jacobian.getRows(); i++) {
            for(int j = 0; j < jacobian.getColumns(); j++) {
                if(i == j && unactivated.getValue(i).getValue() > 0) jacobian.setValue(new NetworkValue(1.0), i, j);
                else jacobian.setValue(new NetworkValue(0.0), i, j);
            }
        }

        return jacobian;
    }
    
}
