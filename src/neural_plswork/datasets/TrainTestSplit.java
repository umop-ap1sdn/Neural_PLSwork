package neural_plswork.datasets;

public class TrainTestSplit {
    
    public static TrainingDataset[] train_test_split(TrainingDataset td, double training_ratio) {
        int train_len = (int)(training_ratio * td.length);
        TrainingDataset[] ret = new TrainingDataset[2];
        ret[0] = td.slice(0, train_len);
        ret[1] = td.slice(train_len, td.length);
        return ret;
    }

    public static BatchedTrainingDataset[] train_test_split(BatchedTrainingDataset btd, double training_ratio) {
        int training_len = (int)(training_ratio * btd.BATCH_NUM);
        BatchedTrainingDataset[] ret = new BatchedTrainingDataset[2];
        ret[0] = btd.slice(0, training_len);
        ret[1] = btd.slice(training_len, btd.BATCH_NUM);
        return ret;
    }
}
