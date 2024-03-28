package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class Softmax implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();

        double maxValue = Double.MIN_VALUE;
        for(NetworkValue n: vector) {
            if(n.getValue() > maxValue) maxValue = n.getValue();
        }

        NetworkValue max = new NetworkValue(-1 * maxValue);

        for(int i = 0; i < vector.getLength(); i++) {
            vector.setValue((NetworkValue) vector.getValue(i).add(max), i);
        }
        
        double divisor = 0;
        for(NetworkValue n: vector) divisor += Math.exp(n.getValue());

        for(NetworkValue n: vector) {
            n.setValue(Math.exp(n.getValue()) / divisor);
        }

        return vector;
    }

    @Override
    public Matrix<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> activated = activate(unactivated);
        
        Matrix<NetworkValue> jacobian = new Matrix<>(unactivated.getLength(), unactivated.getLength());
        for(int i = 0; i < jacobian.getRows(); i++) {
            for(int j = 0; j < jacobian.getColumns(); j++) {
                double value;
                if(i == j) {
                    value = activated.getValue(i).getValue() * (1.0 - activated.getValue(i).getValue());
                } else {
                    value = -1 * activated.getValue(i).getValue() * activated.getValue(j).getValue();
                }

                jacobian.setValue(new NetworkValue(value), i, j);
            }
        }

        return jacobian;
    }
    
}
