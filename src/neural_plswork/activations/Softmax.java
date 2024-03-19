package neural_plswork.activations;

import java.util.Arrays;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Softmax implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        double[] arr = NetworkValue.vectorToArr(unactivated);
        double divisor = 0;
        for(double d: arr) divisor += Math.exp(d);

        for(int i = 0; i < arr.length; i++) {
            arr[i] = Math.exp(arr[i]) / divisor;
        }

        return NetworkValue.arrToVector(arr);
    }

    @Override
    public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> activated = activate(unactivated);
        double[] arr = NetworkValue.vectorToArr(activated);
        double[] derivs = new double[arr.length];
        Arrays.fill(derivs, 0);

        for(int i = 0; i < arr.length; i++) {
            for(int j = 0; j < arr.length; j++) {
                if(i == j) derivs[i] += arr[i] * (1 - arr[i]);
                else derivs[i] -= arr[i] * arr[j];
            }
        }

        return NetworkValue.arrToVector(arr);
    }
    
}
