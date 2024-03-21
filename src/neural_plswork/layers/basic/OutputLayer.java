package neural_plswork.layers.basic;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Vector;

public class OutputLayer extends NeuronLayer {

    public OutputLayer(ActivationFunction activation, int layerSize, int historySize) {
        super(activation, layerSize, historySize, false);
    }

    public Vector<NetworkValue> getOutput() {
        return super.getRecentValues();
    }

    public Vector<NetworkValue> getOutput(int time) {
        return super.getValues(time);
    }

}
