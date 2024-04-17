package neural_plswork.neuron.dropout;

import java.util.Random;

import neural_plswork.core.NetworkValue;
import neural_plswork.math.Vector;

public class SampleDropout implements Dropout {
    private final double p;
    private final Random rand;
    private boolean[] dropout;

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

    @Override
    public void shuffle(int length) {
        dropout = new boolean[length];
        for(int i = 0; i < length; i++) {
            dropout[i] = rand.nextDouble() < p;
        }
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

    public SampleDropout copy() {
        return new SampleDropout(rand, p);
    }
}
