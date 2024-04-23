package neural_plswork.unit.ffUnits;

import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.layers.basic.OutputLayer;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.unit.Unit;

public class OutputUnit extends Unit {
    
    OutputLayer layer;

    public OutputUnit(OutputLayer nLayer, ConnectionLayer[] cLayers, int batchSize, int max_threads) {
        super(new NeuronLayer[]{nLayer}, cLayers, batchSize, max_threads);
        layer = nLayer;
        
    }
    
    @Override
    public void forwardPass(int thread) {
        //Vector<NetworkValue> netSum = cLayers[0].forwardPass(thread);
        Vector<NetworkValue> netSum = null;
        for(ConnectionLayer c: cLayers) {
            if(netSum == null) netSum = c.forwardPass(thread);
            else netSum = netSum.<NetworkValue, NetworkValue>add(c.forwardPass(thread)).getAsVector();
        }

        nLayers[0].activate(netSum, thread);
    }

    public void calcEvals(Vector<NetworkValue> targets, int thread, int time) {
        Vector<NetworkValue> evals = layer.calculateEval(targets, null, 0, thread);
        layer.setEvals(evals, time, thread);
    }

    public Vector<NetworkValue> getOutputs(int time, int thread) {
        return layer.getOutput(time, thread);
    }

    public Vector<NetworkValue> getOutputs(int thread) {
        return layer.getOutput(thread);
    }

    public void setEvaluation(Differentiable eval) {
        layer.setEvaluation(eval);
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

    @Override
    public NeuronLayer[] getExitLayers() {
        return nLayers;
    }

}
