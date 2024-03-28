package neural_plswork.unit.ffUnits;

import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.layers.basic.OutputLayer;
import neural_plswork.math.Vector;
import neural_plswork.unit.Unit;

public class OutputUnit extends Unit {
    
    OutputLayer layer;

    public OutputUnit(OutputLayer nLayer, ConnectionLayer cLayer, int batchSize) {
        super(new NeuronLayer[]{nLayer}, new ConnectionLayer[]{cLayer}, batchSize);
        layer = nLayer;
        
    }
    
    @Override
    public void forwardPass(int thread) {
        Vector<NetworkValue> netSum = cLayers[0].forwardPass(thread);
        nLayers[0].activate(netSum, thread);
    }

    public void calcEvals(Vector<NetworkValue> targets, int thread, int time) {
        Vector<NetworkValue> evals = layer.calculateEval(targets, null, time, thread);
        layer.setEvals(evals, time, thread);
    }

    public Vector<NetworkValue> getOutputs(int time, int thread) {
        return layer.getOutput(time, thread);
    }

    public Vector<NetworkValue> getOutputs(int thread) {
        return layer.getOutput(thread);
    }

    @Override
    public void calcEvals(Unit next, int thread) {
        // Unused
    }

    @Override
    public ConnectionLayer[] getEntryConnections() {
        return cLayers;
    }

    @Override
    public NeuronLayer[] getEntryLayers() {
        return nLayers;
    }

}
