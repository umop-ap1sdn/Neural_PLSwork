package neural_plswork.unit.constructor;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NeuronLayer;
import neural_plswork.initialize.Initializer;
import neural_plswork.network.InvalidNetworkConstructionException;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.regularization.penalize.Penalty;
import neural_plswork.unit.Unit;
import neural_plswork.unit.ffUnits.HiddenUnit;

public class HiddenUnitConstructor implements UnitConstructor {

    @Override
    public Unit construct(NeuronLayer[] prior, ActivationFunction[] activation, int[] layerSize, int historyLength,
            boolean[] bias, int MAX_THREADS, Initializer[] initializer, Penalty[] penalty,
            OptimizationFunction[] optimizer) throws InvalidNetworkConstructionException {
        // HiddenUnit constructor
        if(activation.length != 1 || layerSize.length != 1 || bias.length != 1) 
            throw new InvalidNetworkConstructionException("HiddenUnit can only have 1 layer");
        if(initializer.length != prior.length && optimizer.length != prior.length)
            throw new InvalidNetworkConstructionException("ConnectionLayer parameters must be equal to number of ConnectionLayers");
        
        NeuronLayer hiddenLayer = new NeuronLayer(activation[0], layerSize[0], historyLength, bias[0], MAX_THREADS);

        ConnectionLayer[] cLayers = new ConnectionLayer[prior.length];

        for(int i = 0; i < cLayers.length; i++) {
            cLayers[i] = new ConnectionLayer(prior[i], hiddenLayer, initializer[i], optimizer[i], penalty[i]);
        }

        return new HiddenUnit(hiddenLayer, cLayers, historyLength);
    }
    
}