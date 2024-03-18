package neural_plswork.activations;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Sigmoid implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        double[] arr = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < arr.length; i++) arr[i] = sigmoid(arr[i]);
        return NetworkValue.arrToVector(arr);
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        double[] arr = NetworkValue.vectorToArr(unactivated);
        for(int i = 0; i < arr.length; i++) arr[i] = sigmoid(arr[i]) * (1 - sigmoid(arr[i]));
        return NetworkValue.arrToVector(arr);
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-1 * x));
    }
    
}
