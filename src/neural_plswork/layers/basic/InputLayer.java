package neural_plswork.layers.basic;

import java.util.Arrays;

import neural_plswork.activations.Linear;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class InputLayer extends NeuronLayer {
    
    public InputLayer(int layerSize, int historySize, boolean bias) {
        super(new Linear(), layerSize, historySize, bias);
    }

    public void setInputs(Vector<NetworkValue> inputs) {
        super.activate(inputs);
    }

    @Override
    public Vector<NetworkValue> calculateEval(Vector<NetworkValue> unused0, Matrix<NetworkValue> unused1, int time) {
        double[] errors = new double[super.size()];
        Arrays.fill(errors, 0);
        return NetworkValue.arrToVector(errors);
    }
}
