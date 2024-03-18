package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Tanh implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        double[] arr = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < arr.length; i++) arr[i] = Math.tanh(arr[i]);
        return NetworkValue.arrToVector(arr);
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        double[] arr = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < arr.length; i++) arr[i] = Math.pow(1.0 / Math.cosh(arr[i]), 2);
        return NetworkValue.arrToVector(arr);
    }
    
}
