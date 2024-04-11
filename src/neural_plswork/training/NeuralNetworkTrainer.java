package neural_plswork.training;

import neural_plswork.datasets.TrainingDataset;
import neural_plswork.network.Network;

public abstract class NeuralNetworkTrainer {
    public final Network nn;
    public final TrainingDataset td;

    public NeuralNetworkTrainer(Network nn, TrainingDataset td) {
        this.nn = nn;
        this.td = td;
    }

    public abstract void train_batch();
    public abstract void train_epoch();

    public abstract double computeTrainingEval();
    public abstract double computeValidationEval();
}
