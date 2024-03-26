package neural_plswork.initialize;

public class ConstantInitializer implements Initializer {
    private final double constant;

    public ConstantInitializer(double constant) {
        this.constant = constant;
    }

    public ConstantInitializer() {
        constant = 0;
    }

    public double getNextWeight() {
        return constant;
    }
}
