package neural_plswork.neuron.dropout;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class NoneDropout implements Dropout {

    @Override
    public Vector<NetworkValue> dropout(Vector<NetworkValue> input) {
        return input;
    }

    @Override
    public void shuffle(int length) {
        // Unused
    }

    @Override
    public Dropout copy() {
        return new NoneDropout();
    }
    
}
