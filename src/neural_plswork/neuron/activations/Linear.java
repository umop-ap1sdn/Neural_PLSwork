package neural_plswork.neuron.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class Linear implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        return unactivated.copy();
    }

    @Override
    public Matrix<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Matrix<NetworkValue> jacobian = new Matrix<>(unactivated.getLength(), unactivated.getLength());
        for(int i = 0; i < jacobian.getRows(); i++) {
            for(int j = 0; j < jacobian.getColumns(); j++) {
                if(i == j) jacobian.setValue(new NetworkValue(1.0), i, j);
                else jacobian.setValue(new NetworkValue(0.0), i, j);
            }
        }

        return jacobian;
    }
    
}
