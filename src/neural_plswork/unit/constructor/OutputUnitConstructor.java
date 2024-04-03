package neural_plswork.unit.constructor;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NeuronLayer;
import neural_plswork.evaluation.Differentiable;
import neural_plswork.initialize.Initializer;
import neural_plswork.layers.basic.OutputLayer;
import neural_plswork.network.InvalidNetworkConstructionException;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.regularization.penalize.Penalty;
import neural_plswork.unit.ffUnits.OutputUnit;

public class OutputUnitConstructor implements UnitConstructor {

    private Differentiable eval;

    public OutputUnitConstructor(Differentiable eval) {
        this.eval = eval;
    }

    @Override
    public OutputUnit construct(NeuronLayer[] prior, ActivationFunction[] activation, int[] layerSize,
            boolean[] bias, Initializer[] initializer, Penalty[] penalty,
            OptimizationFunction[] optimizer, int historyLength, int MAX_THREADS) throws InvalidNetworkConstructionException {
        // HiddenUnit constructor
        if(activation.length != 1 || layerSize.length != 1 || bias.length != 0) 
            throw new InvalidNetworkConstructionException("OutputUnit can only have 1 layer and no bias");
        if(initializer.length != prior.length && optimizer.length != prior.length || penalty.length != prior.length)
            throw new InvalidNetworkConstructionException("ConnectionLayer parameters must be equal to number of ConnectionLayers");
        
        OutputLayer hiddenLayer = new OutputLayer(eval, activation[0], layerSize[0], historyLength, MAX_THREADS);

        ConnectionLayer[] cLayers = new ConnectionLayer[prior.length];

        for(int i = 0; i < cLayers.length; i++) {
            cLayers[i] = new ConnectionLayer(prior[i], hiddenLayer, initializer[i], optimizer[i], penalty[i]);
        }

        return new OutputUnit(hiddenLayer, cLayers, historyLength);
    }
    
}