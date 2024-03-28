package neural_plswork.unit;

import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NeuronLayer;

public abstract class Unit {
    final NeuronLayer[] nLayers;
    final ConnectionLayer[] cLayers;

    final int batchSize;

    public Unit(NeuronLayer[] nLayers, ConnectionLayer[] cLayers, int batchSize) {
        this.nLayers = nLayers;
        this.cLayers = cLayers;
        this.batchSize = batchSize;
    }

    public void purgeEval(int thread) {
        for(NeuronLayer n: nLayers) n.purgeEval(thread);
    }

    public void purgeEval(int thread, int times) {
        for(NeuronLayer n: nLayers) n.purgeEval(thread, times);
    }

    public void clear(int thread) {
        for(NeuronLayer n: nLayers) n.clear(thread);
    }

    public void clear() {
        for(NeuronLayer n: nLayers) n.clear();
    }

    public void adjustWeights(double lr, boolean descending, int thread) {
        for(ConnectionLayer c: cLayers) {
            c.adjustWeights(lr, batchSize, descending, thread);
        }
    }
    
    public abstract void forwardPass(int thread);
    public abstract void calcEvals(int thread);

    abstract ConnectionLayer[] getEntryConnections();
}
