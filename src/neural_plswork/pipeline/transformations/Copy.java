package neural_plswork.pipeline.transformations;

import neural_plswork.dataframe.DoubleColumn;

public class Copy implements Transformation {

    @Override
    public void fit(DoubleColumn[] input) {
        // unused
    }

    @Override
    public DoubleColumn[] transform(DoubleColumn[] input) {
        DoubleColumn[] output = new DoubleColumn[input.length];

        for(int i = 0; i < input.length; i++) {
            Double[] column = new Double[input[i].size()];
            for(int j = 0; j < column.length; j++) {
                column[j] = input[i].get(j);
            }

            output[i] = new DoubleColumn(column);
        }

        return output;
    }
    
}
