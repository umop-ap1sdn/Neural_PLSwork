package neural_plswork.pipeline.transformations;

import java.util.Arrays;

import neural_plswork.dataframe.DoubleColumn;

public class Normalize implements Transformation, Reversible {
    private double[] means;
    private double[] vars;

    private final double newMean;
    private final double newVar;

    public Normalize(double mean, double var) {
        newMean = mean;
        newVar = var;
    }

    @Override
    public void fit(DoubleColumn[] input) {
        means = new double[input.length];
        vars = new double[input.length];
        Arrays.fill(means, 0);
        Arrays.fill(vars, 0);

        for(int i = 0; i < input.length; i++) {
            for(double d: input[i]) {
                means[i] += d / input[i].size();
                vars[i] += Math.pow(d, 2) / input[i].size();
            }
        }

        for(int i = 0; i < input.length; i++) {
            vars[i] -= Math.pow(means[i], 2);
        }
    }

    @Override
    public DoubleColumn[] transform(DoubleColumn[] input) {
        if(means == null || input.length != means.length) throw new TransformationException("Must fit before transform can be called");

        DoubleColumn[] output = new DoubleColumn[input.length];
        for(int i = 0; i < input.length; i++) {
            Double[] column = new Double[input[i].size()];
            for(int j = 0; j < column.length; j++) {
                double val = 0;
                if(vars[i] > 0) val = (input[i].get(j) - means[i]) / Math.sqrt(vars[i]);
                val *= Math.sqrt(newVar);
                val += newMean;
                column[j] = val;
            }

            output[i] = new DoubleColumn(column);
        }

        return output;
    }

    @Override
    public DoubleColumn[] reverse(DoubleColumn[] input) {
        if(means == null || input.length != means.length) throw new TransformationException("Must fit before reverse can be called");

        DoubleColumn[] output = new DoubleColumn[input.length];
        for(int i = 0; i < input.length; i++) {
            Double[] column = new Double[input[i].size()];
            for(int j = 0; j < column.length; j++) {
                double val = 0;
                if(newVar > 0) val = (input[i].get(j) - newMean) / Math.sqrt(newVar);
                val *= Math.sqrt(vars[i]);
                val += means[i];
                column[j] = val;
            }

            output[i] = new DoubleColumn(column);
        }

        return output;
    }


}
