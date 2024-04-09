package neural_plswork.datasets;

public class ThreadedTrainingDataset {
    
    private final int BATCH_SIZE;
    private final int MAX_THREADS;

    private final TrainingDataset[] dataset_thread;

    public ThreadedTrainingDataset(TrainingDataset dataset, int BATCH_SIZE, int MAX_THREADS) {
        this.BATCH_SIZE = BATCH_SIZE;
        this.MAX_THREADS = MAX_THREADS;

        this.dataset_thread = new TrainingDataset[MAX_THREADS];
        initialize_threads(dataset);
    }

    private void initialize_threads(TrainingDataset dataset) {
        int threaded_size = dataset.length() / (BATCH_SIZE * MAX_THREADS);
        for(int i = 0; i < MAX_THREADS; i++) {
            dataset_thread[i] = dataset.slice(i * threaded_size, (i + 1) * threaded_size);
        }
    }
}
