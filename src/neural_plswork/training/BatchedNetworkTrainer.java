package neural_plswork.training;

import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainTestSplit;
import neural_plswork.network.Network;

public class BatchedNetworkTrainer extends NeuralNetworkTrainer {

    private static final double DEFAULT_LEARNING_RATIO = 0.8;

    private final BatchedTrainingDataset train_set;
    private final BatchedTrainingDataset test_set;
    private final double learning_ratio;
    private int index = 0;

    public BatchedNetworkTrainer(Network nn, BatchedTrainingDataset btd) {
        super(nn);
        learning_ratio = DEFAULT_LEARNING_RATIO;
        BatchedTrainingDataset[] split = TrainTestSplit.train_test_split(btd, learning_ratio);
        train_set = split[0];
        test_set = split[1];

    }
    

    @Override
    public void train_batch() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'train_batch'");
    }

    @Override
    public void train_epoch() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'train_epoch'");
    }

    @Override
    public double computeTrainingEval() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeTrainingEval'");
    }

    @Override
    public double computeValidationEval() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'computeValidationEval'");
    }
    
}
