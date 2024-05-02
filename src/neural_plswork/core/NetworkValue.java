package neural_plswork.core;

import java.util.HashSet;

import neural_plswork.math.MatrixElement;
import neural_plswork.math.Vector;
import neural_plswork.math.Matrix;
import neural_plswork.math.constants.AdditiveIdentity;
import neural_plswork.math.constants.ConstantElement;
import neural_plswork.math.constants.IdentityElement;
import neural_plswork.math.constants.MultiplicativeIdentity;
import neural_plswork.math.constants.NullElement;

public class NetworkValue implements MatrixElement {
    protected double value = 0.0;

    private static HashSet<Class<? extends MatrixElement>> compatible;

    protected NetworkValue() {
        value = 0.0;
        initializeCompatible();
    }

    public NetworkValue(double value) {
        this.value = value;
        initializeCompatible();
    }

    private static void initializeCompatible() {
        if(compatible != null) return;

        compatible = new HashSet<>();

        compatible.add(NetworkValue.class);
        compatible.add(ConstantElement.class);
        compatible.add(NullElement.class);
        compatible.add(IdentityElement.class);
        compatible.add(MultiplicativeIdentity.class);
        compatible.add(AdditiveIdentity.class);
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
        
        return new Vector<NetworkValue>(netValArr);
    }

    @Override
    public synchronized MatrixElement add(MatrixElement other) {
        if(!(other instanceof NetworkValue)) return other.add(this);
        return new NetworkValue(value + ((NetworkValue)other).value);
    }

    @Override
    public synchronized MatrixElement multiply(MatrixElement other) {
        if(!(other instanceof NetworkValue)) return other.multiply(this);
        return new NetworkValue(value * ((NetworkValue)other).value);
    }

    @Override
    public synchronized MatrixElement copy() {
        return new NetworkValue(value);
    }

    @Override
    public boolean addable(MatrixElement other) {
        return compatible.contains(other.getClass());
    }

    @Override
    public boolean multipliable(MatrixElement other) {
        return compatible.contains(other.getClass());
    }

    @Override
    public String toString() {
        return String.format("%.5f", value);
    }

    public static class NetworkValueParser {

        public static Vector<NetworkValue> stringToVector(String stringVec) {
            NetworkValue[] values = stringToArr(stringVec);
            return new Vector<>(values);
        }

        public static Matrix<NetworkValue> stringToMatrix(String[] stringVecs) {
            NetworkValue[][] values = new NetworkValue[stringVecs.length][];
            for(int i = 0; i < stringVecs.length; i++) {
                values[i] = stringToArr(stringVecs[i]);
            }

            return new Matrix<>(values);
        }

        private static NetworkValue[] stringToArr(String stringVec) {
            String[] split = stringVec.split(",");
            NetworkValue[] values = new NetworkValue[split.length];
            for(int i = 0; i < split.length; i++) values[i] = new NetworkValue(Double.parseDouble(split[i]));
            return values;
        }
    }
}
