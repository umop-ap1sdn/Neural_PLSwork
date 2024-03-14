package neural_plswork.core;

import neural_plswork.math.MatrixElement;
import neural_plswork.math.Vector;
import neural_plswork.math.exceptions.IllegalMatrixException;

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

    public static double[] vectorToArr(Vector<NetworkValue> vector) {
        double[] ret = new double[vector.getLength()];
        
        for(int i = 0; i < ret.length; i++) {
            ret[i] = vector.getValue(i).value;
        }
        
        return ret;
    }

    public static Vector<NetworkValue> arrToVector(double[] arr) {
        NetworkValue[] netValArr = new NetworkValue[arr.length];
        for(int i = 0; i < arr.length; i++) {
            netValArr[i] = new NetworkValue(arr[i]);
        }
        
        try {
            return new Vector<NetworkValue>(netValArr);
        } catch (IllegalMatrixException e) {
            System.err.println("an error occurred.");
            System.err.println(e.getMessage());
            return null;
        }
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
