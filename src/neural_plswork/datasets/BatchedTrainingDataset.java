package neural_plswork.datasets;

import java.util.Iterator;

public class BatchedTrainingDataset implements Iterable<TrainingDataset> {
    
    protected final int BATCH_SIZE;
    protected final int BATCH_NUM;

    protected final TrainingDataset[] dataset_batch;

    public BatchedTrainingDataset(TrainingDataset dataset, int BATCH_SIZE) {
        this.BATCH_SIZE = BATCH_SIZE;
        this.BATCH_NUM = dataset.length() / BATCH_SIZE;

        this.dataset_batch = new TrainingDataset[BATCH_NUM];
        initialize_threads(dataset);
    }

    public BatchedTrainingDataset(TrainingDataset[] btd) {
        this.dataset_batch = btd;
        this.BATCH_NUM = btd.length;
        this.BATCH_SIZE = btd[0].length;
    }

    private void initialize_threads(TrainingDataset dataset) {
        for(int i = 0; i < BATCH_NUM; i++) {
            dataset_batch[i] = dataset.slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE);
        }
    }

    public BatchedTrainingDataset slice(int start, int end) {
        TrainingDataset[] btd = new TrainingDataset[end - start];
        for(int i = start; i < end; i++) {
            btd[i] = dataset_batch[i + start];
        }

        return new BatchedTrainingDataset(btd);
    }

    public TrainingDataset getDataset(int thread) {
        return dataset_batch[thread];
    }

    @Override
    public Iterator<TrainingDataset> iterator() {
        return new BatchedDatasetIterator(this);
    }
}
