package neural_plswork.core;

import java.util.Arrays;

import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.rollingqueue.RollingQueue;

public class NeuronLayer {
    
    private RollingQueue<Vector<NetworkValue>>[] unactivated;
    private RollingQueue<Vector<NetworkValue>>[] activated;
    private RollingQueue<Matrix<NetworkValue>>[] derivative;
    private RollingQueue<Vector<NetworkValue>>[] eval;

    private final ActivationFunction activation;

    private final int layerSize;
    private final int historySize;
    private final boolean bias;

    private final int MAX_THREADS;

    public NeuronLayer(ActivationFunction activation, int layerSize, int historySize, boolean bias, int MAX_THREADS) {
        this.activation = activation;
        this.layerSize = layerSize;
        this.historySize = historySize;
        this.bias = bias;

        this.MAX_THREADS = MAX_THREADS;

        initLists();
    }

    @SuppressWarnings("unchecked")
    private void initLists() {
        unactivated = new RollingQueue[MAX_THREADS];
        activated = new RollingQueue[MAX_THREADS];
        derivative = new RollingQueue[MAX_THREADS];
        eval = new RollingQueue[MAX_THREADS];

        for(int i = 0; i < MAX_THREADS; i++) {
            unactivated[i] = new RollingQueue<Vector<NetworkValue>>(historySize);
            activated[i] = new RollingQueue<Vector<NetworkValue>>(historySize);
            derivative[i] = new RollingQueue<Matrix<NetworkValue>>(historySize);
            eval[i] = new RollingQueue<Vector<NetworkValue>>(historySize);
            purgeEval(i);
        }
    }

    public void clear(int thread) {
        unactivated[thread].clear();
        activated[thread].clear();
        derivative[thread].clear();
        eval[thread].clear();
    }

    public void clear() {
        for(int i = 0; i < MAX_THREADS; i++) {
            clear(i);
        }
    }

    public void activate(Vector<NetworkValue> netSum, int thread) {
        if(unactivated[thread].size() >= historySize) unactivated[thread].pop();
        if(activated[thread].size() >= historySize) activated[thread].pop();
        if(derivative[thread].size() >= historySize) derivative[thread].pop();
        
        unactivated[thread].push(netSum);
        activated[thread].push(activation.activate(netSum));
        derivative[thread].push(activation.derivative(netSum));
        
    }

    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> nextErrs, Matrix<NetworkValue> weightsT, int time, int thread) {
        // WeightsT may be changed to MatrixElement in the future, to allow for identity matrices to be used
        Matrix<NetworkValue> multiplied = weightsT.multiply(nextErrs);
        // Matrix<NetworkValue> pointwise = multiplied.pointwiseMultiply(derivative.get(time));
        multiplied = derivative[thread].get(time).multiply(multiplied);
        return multiplied.getAsVector();
    }

    public void setEvals(Vector<NetworkValue> evals, int time, int thread) {
        eval[thread].set(time, evals);
    }

    public void purgeEval(int thread, int times) {
        double[] empty = new double[layerSize];
        Arrays.fill(empty, 0);
        
        Vector<NetworkValue> zeros = NetworkValue.arrToVector(empty);

        // Check to ensure no shallow copy errors occur here
        for(int i = 0; i < times; i++) {
            if(eval[thread].size() == historySize) eval[thread].pop();
            eval[thread].push(zeros);
        }
    }

    public void purgeEval(int thread) {
        purgeEval(thread, historySize);
    }

    public Vector<NetworkValue> getRecentValues(int thread) {
        return activated[thread].getLast();
    }

    public Matrix<NetworkValue> getRecentDerivative(int thread) {
        return derivative[thread].getLast();
    }

    public Vector<NetworkValue> getValues(int time, int thread) {
        if(activated[thread].size() <= time) return null;
        return activated[thread].get(time);
    }

    public Matrix<NetworkValue> getDerivatives(int time, int thread) {
        if(activated[thread].size() <= time) return null;
        return derivative[thread].get(time);
    }

    public Vector<NetworkValue> getEval(int time, int thread) {
        if(activated[thread].size() <= time) return null;
        return eval[thread].get(time);
    }

    public int size() {
        return layerSize;
    }

    public boolean getBias() {
        return bias;
    }

    public final int HISTORY_SIZE() {
        return this.historySize;
    }

    public final int MAX_THREADS() {
        return this.MAX_THREADS;
    }
}
