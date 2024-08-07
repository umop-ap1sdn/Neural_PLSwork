package neural_plswork.training;

import neural_plswork.core.NetworkValue;
import neural_plswork.datasets.BatchedTrainingDataset;
import neural_plswork.datasets.TrainTestSplit;
import neural_plswork.datasets.TrainingDataset;
import neural_plswork.math.Vector;
import neural_plswork.network.Network;

public class BatchedNetworkTrainer extends NeuralNetworkTrainer {

    private static final double DEFAULT_TRAINING_RATIO = 0.8;

    private final BatchedTrainingDataset train_set;
    private final BatchedTrainingDataset test_set;
    private final double training_ratio;
    private int index = 0;

    public BatchedNetworkTrainer(Network nn, TrainingDataset td) {
        super(nn, td);
        training_ratio = DEFAULT_TRAINING_RATIO;
        BatchedTrainingDataset btd = new BatchedTrainingDataset(td, nn.batch_size());
        BatchedTrainingDataset[] split = TrainTestSplit.train_test_split(btd, training_ratio);
        train_set = split[0];
        test_set = split[1];

    }

    public BatchedNetworkTrainer(Network nn, TrainingDataset td, double training_ratio) {
        super(nn, td);
        this.training_ratio = training_ratio;
        BatchedTrainingDataset btd = new BatchedTrainingDataset(td, nn.batch_size());
        BatchedTrainingDataset[] split = TrainTestSplit.train_test_split(btd, training_ratio);
        train_set = split[0];
        test_set = split[1];

    }


    @Override
    public void train_batch() {
        nn.setDropoutEnable(true);
        if(train_set.batch_num() == 0) return;
        TrainingDataset td = train_set.getDataset(index);
        int batchIndex = 0;
        for(Vector<NetworkValue>[] sample: td) {
            nn.predict(sample[0], 0);
            nn.calcEvals(sample[1], 0, batchIndex++);
        }

        nn.passEvals(0);
        nn.calculateGradients(0);
        nn.adjustWeights(0);
        index = (index + 1) % train_set.batch_num();
    }

    @Override
    public void train_epoch() {
        do {
            train_batch();
        } while(index != 0);
    }

    @Override
    @SuppressWarnings("unchecked")
    public double computeTrainingEval() {
        nn.setDropoutEnable(false);
        
        double eval = 0;
        for(int i = 0; i < train_set.batch_num(); i++) {
            Vector<NetworkValue>[] y_pred = new Vector[nn.batch_size()];
            int y_pred_idx = 0;

            TrainingDataset td = train_set.getDataset(i);
            for(Vector<NetworkValue>[] sample: td) {
                y_pred[y_pred_idx++] = nn.predict(sample[0], 0);
            }

            eval += nn.calculateEval(td.getLabels(), y_pred);

        }

        return eval / train_set.batch_num();
    }

    @Override
    @SuppressWarnings("unchecked")
    public double computeValidationEval() {
        nn.setDropoutEnable(false);

        double eval = 0;
        for(int i = 0; i < test_set.batch_num(); i++) {
            Vector<NetworkValue>[] y_pred = new Vector[nn.batch_size()];
            int y_pred_idx = 0;

            TrainingDataset td = test_set.getDataset(i);
            for(Vector<NetworkValue>[] sample: td) {
                y_pred[y_pred_idx++] = nn.predict(sample[0], 0);
            }

            eval += nn.calculateEval(td.getLabels(), y_pred);

        }

        return eval / test_set.batch_num();
    }
    
}
