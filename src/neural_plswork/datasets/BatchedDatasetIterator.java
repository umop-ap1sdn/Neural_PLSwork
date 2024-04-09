package neural_plswork.datasets;

import java.util.Iterator;

public class BatchedDatasetIterator implements Iterator<TrainingDataset> {
    
    private final BatchedTrainingDataset btd;
    private int index = 0;

    protected BatchedDatasetIterator(BatchedTrainingDataset btd) {
        this.btd = btd;
    }

    public boolean hasNext() {
        return index < btd.BATCH_NUM;
    }

    public TrainingDataset next() {
        return btd.dataset_batch[index++];
    }
}
