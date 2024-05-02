package neural_plswork.connection.initialize;

import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.core.NetworkValue;

public class PredefinedInitializer implements Initializer {
    private final Matrix<NetworkValue> primary;
    private final Vector<NetworkValue> bias;

    public PredefinedInitializer(Matrix<NetworkValue> primary, Vector<NetworkValue> bias) {
        this.primary = primary;
        this.bias = bias;
    }

    @Override
    public NetworkValue getNextWeight(int row, int column) {
        if(column == -1) return bias.getValue(row);
        return primary.getValue(row, column);
    }

    @Override
    public Initializer copy() {
        return new PredefinedInitializer(primary.copy(), bias.copy());
    }

}
