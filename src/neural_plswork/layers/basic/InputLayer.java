package neural_plswork.layers.basic;

import neural_plswork.activations.Linear;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Vector;

public class InputLayer extends NeuronLayer {
    
    public InputLayer(int layerSize, int historySize, boolean bias) {
        super(new Linear(), layerSize, historySize, bias);
    }

    public void setInputs(Vector<NetworkValue> inputs) {
        super.activate(inputs);
    }
}
