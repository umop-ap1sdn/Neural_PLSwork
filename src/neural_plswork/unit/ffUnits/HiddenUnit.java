package neural_plswork.unit.ffUnits;

import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.unit.Unit;

public class HiddenUnit extends Unit {
    
    public HiddenUnit(NeuronLayer nLayer, ConnectionLayer cLayer, int batchSize) {
        super(new NeuronLayer[]{nLayer}, new ConnectionLayer[]{cLayer}, batchSize);
        
    }
    
    @Override
    public void forwardPass(int thread) {
        Vector<NetworkValue> netSum = cLayers[0].forwardPass(thread);
        nLayers[0].activate(netSum, thread);
    }

    @Override
    @SuppressWarnings("unchecked")
    public void calcEvals(Unit next, int thread) {
        Matrix<NetworkValue>[] evalMats = new Matrix[next.getEntryConnections().length];
        for(int i = 0; i < evalMats.length; i++) evalMats[i] = next.getEntryConnections()[i].getLayer().transpose();

        for(int i = 0; i < batchSize; i++) {
            Vector<NetworkValue> eval = nLayers[0].calculateEval(next.getEntryLayers()[0].getEval(i, thread), evalMats[0], i, thread);
            for(int j = 1; j < evalMats.length; j++) {
                eval = eval.<NetworkValue, NetworkValue>add(nLayers[0].calculateEval(next.getEntryLayers()[j].getEval(i, thread), evalMats[j], i, thread)).getAsVector();
            }

            nLayers[0].setEvals(eval, i, thread);
        }
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
