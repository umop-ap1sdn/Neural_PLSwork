package neural_plswork.neuron.dropout;

import java.util.Random;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

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

    @Override
    public void shuffle(int length) {
        if(iter == 0) {
            dropout = new boolean[length];
            for(int i = 0; i < length; i++) {
                dropout[i] = rand.nextDouble() < p;
            }
        }

        iter = (iter + 1) % batch_size;
    }

    public Vector<NetworkValue> dropout(Vector<NetworkValue> input) throws InvalidDropoutException {
        if(dropout == null || dropout.length != input.getLength()) throw new InvalidDropoutException("Must call shuffle before dropout can be performed");
        
        int count = 0;
        for(boolean b: dropout) if(b) count++;
        double ratio = 1.0 / (1.0 - ((double)count / dropout.length));

        NetworkValue[] dropped = new NetworkValue[input.getLength()];
        for(int i = 0; i < input.getLength(); i++) {
            if(dropout[i]) dropped[i] = new NetworkValue(0);
            else dropped[i] = new NetworkValue(input.getValue(i).getValue() * ratio);
        }

        return new Vector<>(dropped);
    }

    public BatchDropout copy() {
        return new BatchDropout(rand, p, batch_size);
    }
}
