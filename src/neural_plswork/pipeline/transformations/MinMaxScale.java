package neural_plswork.pipeline.transformations;

import neural_plswork.dataframe.DoubleColumn;

public class MinMaxScale implements Transformation, Reversible {

    private double[] mins;
    private double[] maxs;

    private final double newMin;
    private final double newMax;

    public MinMaxScale(double min, double max) {
        newMin = min;
        newMax = max;
    }

    @Override
    public void fit(DoubleColumn[] input) {
        mins = new double[input.length];
        maxs = new double[input.length];
        
        for(int i = 0; i < input.length; i++) {
            mins[i] = input[i].get(0);
            maxs[i] = input[i].get(0);
            for(int j = 1; j < input[i].size(); j++) {
                if(input[i].get(j) < mins[i]) mins[i] = input[i].get(j);
                if(input[i].get(j) > maxs[i]) maxs[i] = input[i].get(j);
            }
        }
    }

    @Override
    public DoubleColumn[] transform(DoubleColumn[] input) {
        if(mins == null || input.length != mins.length) throw new TransformationException("Must fit before transform can be called");

        DoubleColumn[] output = new DoubleColumn[input.length];
        for(int i = 0; i < input.length; i++) {
            Double[] column = new Double[input[i].size()];
            for(int j = 0; j < column.length; j++) {
                double val = 0;
                if(maxs[i] != mins[i]) val = (input[i].get(j) - mins[i]) / (maxs[i] - mins[i]);
                val *= (newMax - newMin);
                val += newMin;
                column[j] = val;
            }

            output[i] = new DoubleColumn(column);
        }

        return output;
    }

    @Override
    public DoubleColumn[] reverse(DoubleColumn[] input) {
        if(mins == null || input.length != mins.length) throw new TransformationException("Must fit before reverse can be called");

        DoubleColumn[] output = new DoubleColumn[input.length];
        for(int i = 0; i < input.length; i++) {
            Double[] column = new Double[input[i].size()];
            for(int j = 0; j < column.length; j++) {
                double val = 0;
                if(newMax != newMin) val = (input[i].get(j) - newMin) / (newMax - newMin);
                val *= (maxs[i] - mins[i]);
                val += mins[i];
                column[j] = val;
            }

            output[i] = new DoubleColumn(column);
        }

        return output;
    }
    
}
