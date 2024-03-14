package neural_plswork.core;

import neural_plswork.math.MatrixElement;

public class NetworkValue implements MatrixElement {
    protected double value = 0.0;

    protected NetworkValue() {
        value = 0.0;
    }

    public NetworkValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    @Override
    public MatrixElement add(MatrixElement other) {
        if(!(other instanceof NetworkValue)) return other.add(this);
        return new NetworkValue(value + ((NetworkValue)other).value);
    }

    @Override
    public MatrixElement multiply(MatrixElement other) {
        if(!(other instanceof NetworkValue)) return other.multiply(this);
        return new NetworkValue(value * ((NetworkValue)other).value);
    }

    @Override
    public MatrixElement negate() {
        return new NetworkValue(-1 * value);
    }

    @Override
    public MatrixElement copy() {
        return new NetworkValue(value);
    }
}
