package neural_plswork.activations;

import java.util.Arrays;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class Softmax implements ActivationFunction {

    @Override
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated) {
        Vector<NetworkValue> vector = (Vector<NetworkValue>) unactivated.copy();
        
        double divisor = 0;
        for(NetworkValue n: vector) divisor += Math.exp(n.getValue());

        for(NetworkValue n: vector) {
            n.setValue(Math.exp(n.getValue()) / divisor);
        }

        return vector;
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

            System.out.println(derivs[i]);
        }

        return NetworkValue.arrToVector(derivs);
    }
    
}
