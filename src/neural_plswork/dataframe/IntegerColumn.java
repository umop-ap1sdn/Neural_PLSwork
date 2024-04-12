package neural_plswork.dataframe;

public class IntegerColumn extends Column<Integer> {

    public IntegerColumn(Integer[] data) {
        super(data);
    }

    @Override
    public DoubleColumn numify() {
        Double[] ret = new Double[data.length];

        for(int i = 0; i < ret.length; i++) {
            ret[i] = Double.valueOf(data[i]);
        }

        return new DoubleColumn(ret);
    }

    @Override
    public IntegerColumn fillEmpty(Integer replace) {
        Integer[] ret = new Integer[data.length];
        for(int i = 0; i < ret.length; i++) {
            if(data[i] == null) ret[i] = replace;
            else ret[i] = data[i];
        }

        return new IntegerColumn(ret);
    }
    
}
