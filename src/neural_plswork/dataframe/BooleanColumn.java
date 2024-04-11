package neural_plswork.dataframe;

public class BooleanColumn extends Column<Boolean> {

	public BooleanColumn(Boolean[] data) {
		super(data);
	}
    
    @Override
    public DoubleColumn numify() {
        Double[] arr = new Double[data.length];
        for(int i = 0; i < data.length; i++) {
            if(data[i] == null) arr[i] = Double.NaN;
            else if(data[i].equals(true)) arr[i] = 1.0;
            else arr[i] = 0.0;
        }

        return new DoubleColumn(arr);
    }
}
