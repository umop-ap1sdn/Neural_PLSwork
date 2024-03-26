package neural_plswork.initialize;

import java.util.Random;

public class NormalRandomInitializer implements Initializer {
    private final Random rand;
    private final double mean;
    private final double variance;
    private final double sigma;

    public NormalRandomInitializer(Random rand, double mean, double variance) {
        this.rand = rand;
        this.mean = mean;
        this.variance = variance;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer(long seed, double mean, double variance) {
        this.rand = new Random(seed);
        this.mean = mean;
        this.variance = variance;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer(double mean, double variance) {
        this.rand = new Random();
        this.mean = mean;
        this.variance = variance;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer(Random rand) {
        this.rand = rand;
        this.mean = 0;
        this.variance = 0.1;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer(long seed) {
        this.rand = new Random(seed);
        this.mean = 0.0;
        this.variance = 0.1;
        sigma = Math.sqrt(variance);
    }

    public NormalRandomInitializer() {
        this.rand = new Random();
        this.mean = 0.0;
        this.variance = 0.1;
        sigma = Math.sqrt(variance);
    }

    @Override
    public double getNextWeight() {
        double random = rand.nextGaussian();
        random -= mean;
        random *= sigma;
        return random;
    }
}
