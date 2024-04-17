package neural_plswork.layers.basic;

import java.util.Arrays;

import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.neuron.activations.Linear;
import neural_plswork.neuron.dropout.NoneDropout;

public class InputLayer extends NeuronLayer {
    
    public InputLayer(int layerSize, int historySize, boolean bias, int MAX_THREADS) {
        super(new Linear(), new NoneDropout(), layerSize, historySize, bias, MAX_THREADS);
    }

    public void setInputs(Vector<NetworkValue> inputs, int thread) {
        super.activate(inputs, thread);
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> unused0, Matrix<NetworkValue> unused1, int time, int thread) {
        double[] errors = new double[super.size()];
        Arrays.fill(errors, 0);
        return NetworkValue.arrToVector(errors);
    }
}
