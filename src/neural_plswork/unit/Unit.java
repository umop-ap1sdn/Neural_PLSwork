package neural_plswork.unit;

import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NeuronLayer;

public class Unit {
    private final NeuronLayer[] nLayers;
    private final ConnectionLayer[] cLayers;
    
    private final int MAX_THREADS;

    public Unit(NeuronLayer nLayer, ConnectionLayer cLayer, int MAX_THREADS) {
        nLayers = new NeuronLayer[]{nLayer};
        cLayers = new ConnectionLayer[]{cLayer};
        this.MAX_THREADS = MAX_THREADS;
    }
}
