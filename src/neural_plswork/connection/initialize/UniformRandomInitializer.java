package neural_plswork.connection.initialize;

import java.util.Random;

import neural_plswork.core.NetworkValue;

public class UniformRandomInitializer implements Initializer {
    private final Random rand;
    private final double min;
    private final double max;

    public UniformRandomInitializer(Random rand, double min, double max) {
        this.rand = rand;
        this.min = min;
        this.max = max;
    }

    public UniformRandomInitializer(long seed, double min, double max) {
        this.rand = new Random(seed);
        this.min = min;
        this.max = max;
    }

    public UniformRandomInitializer(double min, double max) {
        this.rand = new Random();
        this.min = min;
        this.max = max;
    }

    public UniformRandomInitializer(Random rand) {
        this.rand = rand;
        this.min = -0.1;
        this.max = 0.1;
    }

    public UniformRandomInitializer(long seed) {
        this.rand = new Random(seed);
        this.min = -0.1;
        this.max = 0.1;
    }

    public UniformRandomInitializer() {
        this.rand = new Random();
        this.min = -0.1;
        this.max = 0.1;
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
