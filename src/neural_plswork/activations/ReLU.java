package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class ReLU implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        double[] unacArray = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < unacArray.length; i++) unacArray[i] = Math.max(unacArray[i], 0);

        return NetworkValue.arrToVector(unacArray);
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        double[] unacArray = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < unacArray.length; i++) unacArray[i] = (unacArray[i] > 0) ? 1.0 : 0.0;

        return NetworkValue.arrToVector(unacArray);
    }
    
}
