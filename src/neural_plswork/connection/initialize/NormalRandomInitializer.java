package neural_plswork.connection.initialize;

import java.util.Random;

import neural_plswork.core.NetworkValue;

public class NormalRandomInitializer implements Initializer {
    private final Random rand;
    private final double mean;
    private final double variance;
    private final double sigma;

    private static final double DEFAULT_MEAN = 0.0;
    private static final double DEFAULT_VAR = 0.5;

    public NormalRandomInitializer(Random rand, double mean, double variance) {
        this.rand = rand;
        this.mean = mean;
        this.variance = variance;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer(long seed, double mean, double variance) {
        this(new Random(seed), mean, variance);
    }

    public NormalRandomInitializer(double mean, double variance) {
        this(new Random(), mean, variance);
    }

    public NormalRandomInitializer(Random rand) {
        this(rand, DEFAULT_MEAN, DEFAULT_VAR);
    }

    public NormalRandomInitializer(long seed) {
        this(new Random(seed), DEFAULT_MEAN, DEFAULT_VAR);
    }

    public NormalRandomInitializer() {
        this(new Random(), DEFAULT_MEAN, DEFAULT_VAR);
    }

    @Override
    public NetworkValue getNextWeight(int row, int col) {
        double random = rand.nextGaussian();
        random -= mean;
        random *= sigma;
        return new NetworkValue(random);
    }

    @Override
    public NormalRandomInitializer copy() {
        return new NormalRandomInitializer(rand, mean, variance);
    }
}
