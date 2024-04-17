package neural_plswork.neuron.dropout;

import java.util.Random;

public class SampleDropout implements Dropout {
    private final double p;
    private final Random rand;

    public SampleDropout(double p) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = new Random();
        this.p = p;
    }

    public SampleDropout(long seed, double p) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = new Random(seed);
        this.p = p;
    }

    public SampleDropout(Random rand, double p) {
        if(p < 0.0 || p > 1.0) throw new IllegalArgumentException("p parameter must be between 0 and 1");
        this.rand = rand;
        this.p = p;
    }

    public boolean[] dropout(int length) {
        boolean[] ret = new boolean[length];
        for(int i = 0; i < length; i++) {
            ret[i] = rand.nextDouble() > p;
        }
        
        return ret;
    }

    public SampleDropout copy() {
        return new SampleDropout(rand, p);
    }
}
