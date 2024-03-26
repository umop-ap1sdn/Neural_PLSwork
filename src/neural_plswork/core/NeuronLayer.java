package neural_plswork.core;

import java.util.Arrays;
import java.util.LinkedList;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.math.Matrix;
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

    public Vector<NetworkValue> calculateError(Vector<NetworkValue> nextErrs, Matrix<NetworkValue> weightsT, int time) {
        // WeightsT may be changed to MatrixElement in the future, to allow for identity matrices to be used
        Matrix<NetworkValue> multiplied = weightsT.multiply(nextErrs);
        Matrix<NetworkValue> pointwise = multiplied.pointwiseMultiply(derivative.get(time));
        return pointwise.getAsVector();
    }

    public void setErrors(Vector<NetworkValue> errors, int time) {
        error.set(time, errors);
    }

    public void purgeErrors(int times) {
        double[] empty = new double[layerSize];
        Arrays.fill(empty, 0);
        
        Vector<NetworkValue> zeros = NetworkValue.arrToVector(empty);

        // Check to ensure no shallow copy errors occur here
        for(int i = 0; i < times; i++) {
            error.addLast(zeros);
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

    public int size() {
        return layerSize;
    }

    public boolean getBias() {
        return bias;
    }
}
