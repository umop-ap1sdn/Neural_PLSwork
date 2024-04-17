package neural_plswork.neuron.dropout;

import java.util.Random;

public class BatchDropout implements Dropout {
    private final double p;
    private final int batch_size;
    private int iter = 0;
    private final Random rand;

    boolean[] dropout;

    public BatchDropout(double p, int batch_size) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = new Random();
        this.p = p;
        this.batch_size = batch_size;
    }

    public BatchDropout(long seed, double p, int batch_size) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = new Random(seed);
        this.p = p;
        this.batch_size = batch_size;
    }

    public BatchDropout(Random rand, double p, int batch_size) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = rand;
        this.p = p;
        this.batch_size = batch_size;
    }

    private void shuffle(int length) {
        dropout = new boolean[length];
        for(int i = 0; i < length; i++) {
            dropout[i] = rand.nextDouble() > p;
        }
    }

    public boolean[] dropout(int length) {
        if(iter == 0) shuffle(length);
        iter = (iter + 1) % batch_size;

        return dropout;
    }

    public BatchDropout copy() {
        return new BatchDropout(rand, p, batch_size);
    }
}
