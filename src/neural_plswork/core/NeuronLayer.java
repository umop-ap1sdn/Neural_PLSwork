package neural_plswork.core;

import java.util.Arrays;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.rollingqueue.RollingQueue;

public class NeuronLayer {
    
    private RollingQueue<Vector<NetworkValue>> unactivated;
    private RollingQueue<Vector<NetworkValue>> activated;
    private RollingQueue<Matrix<NetworkValue>> derivative;
    private RollingQueue<Vector<NetworkValue>> eval;

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
        unactivated = new RollingQueue<>(historySize);
        activated = new RollingQueue<>(historySize);
        derivative = new RollingQueue<>(historySize);
        eval = new RollingQueue<>(historySize);
    }

    public void activate(Vector<NetworkValue> netSum) {
        if(unactivated.size() >= historySize) unactivated.pop();
        if(activated.size() >= historySize) activated.pop();
        if(derivative.size() >= historySize) derivative.pop();
        
        unactivated.push(netSum);
        activated.push(activation.activate(netSum));
        derivative.push(activation.derivative(netSum));

        
        
    }

    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> nextErrs, Matrix<NetworkValue> weightsT, int time) {
        // WeightsT may be changed to MatrixElement in the future, to allow for identity matrices to be used
        Matrix<NetworkValue> multiplied = weightsT.multiply(nextErrs);
        // Matrix<NetworkValue> pointwise = multiplied.pointwiseMultiply(derivative.get(time));
        multiplied = derivative.get(time).multiply(multiplied);
        return multiplied.getAsVector();
    }

    public void setEvals(Vector<NetworkValue> evals, int time) {
        eval.set(time, evals);
    }

    public void purgeEval(int times) {
        double[] empty = new double[layerSize];
        Arrays.fill(empty, 0);
        
        Vector<NetworkValue> zeros = NetworkValue.arrToVector(empty);

        // Check to ensure no shallow copy errors occur here
        for(int i = 0; i < times; i++) {
            eval.pop();
            eval.push(zeros);
        }
    }

    public Vector<NetworkValue> getRecentValues() {
        return activated.getLast();
    }

    public Matrix<NetworkValue> getRecentDerivative() {
        return derivative.getLast();
    }

    public Vector<NetworkValue> getValues(int time) {
        return activated.get(time);
    }

    public Matrix<NetworkValue> getDerivatives(int time) {
        return derivative.get(time);
    }

    public Vector<NetworkValue> getEval(int time) {
        return eval.get(time);
    }

    public int size() {
        return layerSize;
    }

    public boolean getBias() {
        return bias;
    }
}
