package neural_plswork.training;

import neural_plswork.network.Network;

public abstract class NeuralNetworkTrainer {
    public final Network nn;

    public NeuralNetworkTrainer(Network nn) {
        this.nn = nn;
    }

    public abstract void train_batch();
    public abstract void train_epoch();

    public abstract double computeTrainingEval();
    public abstract double computeValidationEval();
}
