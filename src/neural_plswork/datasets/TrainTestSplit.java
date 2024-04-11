package neural_plswork.datasets;

public class TrainTestSplit {
    
    public static TrainingDataset[] train_test_split(TrainingDataset td, double training_ratio) {
        if(training_ratio < 0.0 || training_ratio > 1.0) throw new IllegalArgumentException("Training ratio must be between 0 and 1");
        
        int train_len = (int)(training_ratio * td.length);
        TrainingDataset[] ret = new TrainingDataset[2];
        ret[0] = td.slice(0, train_len);
        ret[1] = td.slice(train_len, td.length);
        return ret;
    }

    public static BatchedTrainingDataset[] train_test_split(BatchedTrainingDataset btd, double training_ratio) {
        if(training_ratio < 0.0 || training_ratio > 1.0) throw new IllegalArgumentException("Training ratio must be between 0 and 1");

        int train_len = (int)(training_ratio * btd.BATCH_NUM);
        BatchedTrainingDataset[] ret = new BatchedTrainingDataset[2];
        ret[0] = btd.slice(0, train_len);
        ret[1] = btd.slice(train_len, btd.BATCH_NUM);
        return ret;
    }
}
