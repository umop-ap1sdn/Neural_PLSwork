package neural_plswork.connection.initialize;

import java.util.Random;

import neural_plswork.core.NetworkValue;

public class UniformRandomInitializer implements Initializer {
    private final Random rand;
    private final double min;
    private final double max;

    private static final double DEFAULT_MIN = 0.3;
    private static final double DEFAULT_MAX = 0.3;

    public UniformRandomInitializer(Random rand, double min, double max) {
        this.rand = rand;
        this.min = min;
        this.max = max;
    }

    public UniformRandomInitializer(long seed, double min, double max) {
        this(new Random(seed), min, max);
    }

    public UniformRandomInitializer(double min, double max) {
        this(new Random(), min, max);
    }

    public UniformRandomInitializer(Random rand) {
        this(rand, DEFAULT_MIN, DEFAULT_MAX);
    }

    public UniformRandomInitializer(long seed) {
        this(new Random(seed), DEFAULT_MIN, DEFAULT_MAX);
    }

    public UniformRandomInitializer() {
        this(new Random(), DEFAULT_MIN, DEFAULT_MAX);
    }

    

    @Override
    public NetworkValue getNextWeight(int row, int col) {
        double random = rand.nextDouble();
        random *= (max - min);
        random += min;

        return new NetworkValue(random);
    }

    @Override
    public UniformRandomInitializer copy() {
        return new UniformRandomInitializer(rand, min, max);
    }
}
