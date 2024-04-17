package neural_plswork.unit.constructor;

import neural_plswork.network.InvalidNetworkConstructionException;
import neural_plswork.neuron.activations.ActivationFunction;
import neural_plswork.neuron.dropout.Dropout;
import neural_plswork.connection.initialize.Initializer;
import neural_plswork.connection.optimizer.OptimizationFunction;
import neural_plswork.connection.penalize.Penalty;
import neural_plswork.core.NeuronLayer;
import neural_plswork.unit.Unit;

public interface UnitConstructor {
    public Unit construct(NeuronLayer[] prior, ActivationFunction[] activation, Dropout[] dropout, Integer[] layerSize, 
            Boolean[] bias, Initializer[] initializer, Penalty[] penalty,
            OptimizationFunction[] optimizer, int historyLength, int MAX_THREADS) throws InvalidNetworkConstructionException;
}
