package neural_plswork.dataframe;

public class NumericalDataFrame extends DataFrame {
    private final double[][] dataPrim;
    
	public NumericalDataFrame(String[] labels, DoubleColumn[] data) {
		super(labels, data);
        dataPrim = new double[rows][columns];
        initArray(data);
	}

    private void initArray(DoubleColumn[] data) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                dataPrim[i][j] = data[j].get(i);
            }
        }
    }

    public double[][] getNumData() {
        return dataPrim;
    }
}
