package neural_plswork.dataframe;

public class StringColumn extends Column<String> {

	public StringColumn(String[] data) {
		super(data);
	}
    
    @Override
    public StringColumn fillEmpty(String replace) {
        String[] arr = new String[data.length];
        for(int i = 0; i < arr.length; i++) {
            if(data[i] == null || data[i].equals("")) arr[i] = replace;
            else arr[i] = data[i];
        }

        return new StringColumn(arr);
    }
}
