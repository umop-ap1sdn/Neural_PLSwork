package neural_plswork.unit.constructor;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.network.InvalidNetworkConstructionException;
import neural_plswork.core.NeuronLayer;
import neural_plswork.initialize.Initializer;
import neural_plswork.optimizer.OptimizationFunction;
import neural_plswork.regularization.penalize.Penalty;
import neural_plswork.unit.Unit;

public interface UnitConstructor {
    public Unit construct(NeuronLayer[] prior, ActivationFunction[] activation, int[] layerSize, int historyLength, 
            boolean[] bias, int MAX_THREADS, Initializer[] initializer, Penalty[] penalty,
            OptimizationFunction[] optimizer) throws InvalidNetworkConstructionException;
}
