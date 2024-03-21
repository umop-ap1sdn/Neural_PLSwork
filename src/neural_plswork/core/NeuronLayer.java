package neural_plswork.core;

import java.util.Arrays;
import java.util.LinkedList;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.math.Vector;

public class NeuronLayer {
    
    private LinkedList<Vector<NetworkValue>> unactivated;
    private LinkedList<Vector<NetworkValue>> activated;
    private LinkedList<Vector<NetworkValue>> derivative;
    private LinkedList<Vector<NetworkValue>> error;

    private final ActivationFunction activation;

    private final int layerSize;
    private final int historySize;
    private final boolean bias;

    public NeuronLayer(ActivationFunction activation, int layerSize, int historySize, boolean bias) {
        this.activation = activation;
        this.layerSize = layerSize;
        this.historySize = historySize;
        this.bias = bias;

        initLists();
    }

    private void initLists() {

        unactivated = new LinkedList<>();
        activated = new LinkedList<>();
        derivative = new LinkedList<>();
        error = new LinkedList<>();

        for(int i = 0; i < historySize; i++) {
            double[] initialVector = new double[layerSize];
            Arrays.fill(initialVector, 0);
            unactivated.addLast(NetworkValue.arrToVector(initialVector));
            activated.addLast(NetworkValue.arrToVector(initialVector));
            derivative.addLast(NetworkValue.arrToVector(initialVector));
            error.addLast(NetworkValue.arrToVector(initialVector));
            
        }
    }

    public void activate(Vector<NetworkValue> netSum) {
        unactivated.addLast(netSum);
        activated.addLast(activation.activate(netSum));
        derivative.addLast(activation.derivative(netSum));

        if(unactivated.size() > historySize) unactivated.pollFirst();
        if(activated.size() > historySize) activated.pollFirst();
        if(derivative.size() > historySize) derivative.pollFirst();
        
    }

    public void setErrors(Vector<NetworkValue> errors, int time) {
        error.set(time, errors);
    }

    public void purgeErrors(int times) {
        for(int i = 0; i < times; i++) {
            double[] empty = new double[layerSize];
            Arrays.fill(empty, 0);
            error.addLast(NetworkValue.arrToVector(empty));
            error.pollFirst();
        }
    }

    public Vector<NetworkValue> getRecentValues() {
        return activated.getLast();
    }

    public Vector<NetworkValue> getRecentDerivative() {
        return derivative.getLast();
    }

    public Vector<NetworkValue> getValues(int time) {
        return activated.get(time);
    }

    public Vector<NetworkValue> getDerivatives(int time) {
        return derivative.get(time);
    }

    public Vector<NetworkValue> getError(int time) {
        return error.get(time);
    }

    public boolean getBias() {
        return bias;
    }
}
