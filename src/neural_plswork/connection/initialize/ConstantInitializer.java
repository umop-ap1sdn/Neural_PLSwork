package neural_plswork.connection.initialize;

import neural_plswork.core.NetworkValue;

public class ConstantInitializer implements Initializer {
    private final double constant;

    public ConstantInitializer(double constant) {
        this.constant = constant;
    }

    public ConstantInitializer() {
        constant = 0;
    }

    @Override
    public NetworkValue getNextWeight(int row, int col) {
        return new NetworkValue(constant);
    }

    @Override
    public ConstantInitializer copy() {
        return new ConstantInitializer(constant);
    }
}
