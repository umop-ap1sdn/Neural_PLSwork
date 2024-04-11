package neural_plswork.dataframe;

import java.util.HashMap;

public class DoubleColumn extends Column<Double> {

	public DoubleColumn(Double[] data) {
		super(data);
	}

    @Override
    public DoubleColumn fillEmpty(Double replace) {
        Double[] arr = new Double[data.length];
        for(int i = 0; i < data.length; i++) {
            if(data[i] == null || data[i].isNaN() || data[i].isInfinite()) arr[i] = replace;
            else arr[i] = data[i];
        }

        return new DoubleColumn(arr);
    }
    
    @Override
    public DoubleColumn numify() {
        return this;
    }

    @Override
    public DoubleColumn numify(HashMap<Double, Double> presetMap) {
        return this;
    }
}
