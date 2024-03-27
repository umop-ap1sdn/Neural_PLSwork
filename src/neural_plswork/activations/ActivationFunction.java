package neural_plswork.activations;

import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.core.NetworkValue;

public interface ActivationFunction {
    
    public Vector<NetworkValue> activate(Vector<NetworkValue> unactivated);
    // public Vector<NetworkValue> derivative(Vector<NetworkValue> unactivated);

    // Change all functions to now create a jacobian matrix form derivative, mainly due to softmax activation
    public Matrix<NetworkValue> derivative(Vector<NetworkValue> unactivated);


    public static ActivationFunction getFunction(Activation activation) throws InvalidActivationException {
        if(activation == null) throw new InvalidActivationException("Activation enum is null");
        switch(activation) {
            case CUSTOM: return null;
            case LINEAR: return new Linear();
            case RELU: return new ReLU();
            case SIGMOID: return new Sigmoid();
            case TANH: return new Tanh();
            case SOFTMAX: return new Softmax();
            case INVALID: throw new InvalidActivationException();
        }

        return null;
    }
}
